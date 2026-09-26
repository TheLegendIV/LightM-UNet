"""Per-layer joint bits + folding MILP for FINN dataflow deployment of S12 ENet.

Chooses, per layer, (weight_bits, act_bits) and (PE, SIMD, ram_style, variant),
plus PE/ram_style for every residual-join threshold node, minimizing
    alpha * mean(normalized HAWQ sensitivity) + (1 - alpha) * mean(normalized cycles)
under hard LUT / BRAM_18K / DSP / URAM budgets on xczu7ev and an optional
latency cap. Solved with CBC (pulp).

AGENTS: read MILP/finn_milp.md before changing this file, and update it with
any behavior change -- it holds the formulation, the rationale and evidence
behind every constraint and constant, the known gaps, and the output schema.
Cost formulas live in finn_cost_model.py (+ finn_cost_model.md).

Usage:
    python MILP/finn_milp.py --config config_12_dense_relu_nearest_conv_upsample \\
        --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_conv_upsample.json \\
        --candidate-bits 4,6,8 --alpha 1.0 --force-dsp \\
        --hard-lut-fraction 0.5 --hard-bram-fraction 0.5 --hard-dsp-fraction 0.9 --max-latency-ms 200 \\
        --out-file MILP/artifacts/<dir>/layer_bits_folding_<...>.json
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pulp
import torch  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_utils import enumerate_blocks, path_to_block_map  # noqa: E402
from finn_cost_model import (  # noqa: E402
    IMPL_STYLE_HLS, IMPL_STYLE_RTL, RAM_STYLE_BLOCK, RAM_STYLE_ULTRA, LayerGeometry, calibrated_bram18k,
    calibrated_lut, divisors, layer_cost_pe_simd, max_pe, max_simd, threshold_node_cost,
    FOLDABLE_STREAM_KINDS, STREAM_NODE_KINDS, stream_node_cost,
)
from layer_topology import (  # noqa: E402
    ACT_SUFFIX, CONCAT_SUFFIX, OUT_ACT_SUFFIX, SKIP_PAD_SUFFIX, SKIP_QUANT_SUFFIX, compute_dataflow_graph,
    compute_predecessor_map,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from expand_layer_bits import resolve_act_sources  # noqa: E402
from milp_outputs import write_outputs  # noqa: E402

# ---- Device, search space, CLI-set globals (see finn_milp.md "Constants") ----

XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624, "DSP": 1_728, "URAM": 96}
CANDIDATE_BITS = (2, 4, 6, 8, 16)
INPUT_HW = (512, 512)
RAM_STYLES = (RAM_STYLE_BLOCK, "distributed")  # the standalone Thresholding node's memory -- the only free RAM choice
FORCE_SERIAL = False

VARIANT_RTL_DSP_NOACT1 = "rtl_dsp_noact1"
VARIANT_HLS_LUT_NOACT0 = "hls_lut_noact0"

VARIANT_HLS_DSP_NOACT0 = "hls_dsp_noact0"  # placeholder, never eligible -- see finn_milp.md before enabling

ALLOW_LUT_MULT = False


def _variant_cost_kwargs(variant: str, force_dsp: bool) -> dict:
    """variant -> (impl_style, force_dsp, no_activation) for layer_cost_pe_simd."""
    if variant == VARIANT_RTL_DSP_NOACT1:
        return {"impl_style": IMPL_STYLE_RTL, "force_dsp": force_dsp, "no_activation": True}
    if variant == VARIANT_HLS_LUT_NOACT0:
        return {"impl_style": IMPL_STYLE_HLS, "force_dsp": False, "no_activation": False}
    if variant == VARIANT_HLS_DSP_NOACT0:
        return {"impl_style": IMPL_STYLE_HLS, "force_dsp": True, "no_activation": False}
    raise ValueError(f"unknown resource variant {variant!r}")


def _calibration_force_dsp(variant_kwargs: dict) -> bool:
    """RTL is always DSP, whatever --force-dsp says -- calibrated_lut must know."""
    return variant_kwargs["force_dsp"] or variant_kwargs["impl_style"] == IMPL_STYLE_RTL


def load_config(config_module: str) -> None:
    """Inject a MILP/configs/config_*.py's constants into this module's globals."""
    cfg = importlib.import_module(f"configs.{config_module}")
    globals().update({k: v for k, v in vars(cfg).items() if not k.startswith("_")})


# ---- Model tracing ----

def _pair(v) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else tuple(v)


def trace_layer_geometry(model: torch.nn.Module, input_hw: tuple[int, int], in_channels: int) -> tuple[list[LayerGeometry], list[str]]:
    """Shape of every Conv2d/ConvTranspose2d/MaxPool2d, tagged with its block."""
    blocks = enumerate_blocks(model)
    path_to_block = path_to_block_map(blocks)
    geometries: list[LayerGeometry] = []

    def make_hook(name: str, block_name: str, op_type: str):
        def hook(module, inputs, output):
            x = inputs[0]
            if isinstance(output, tuple):
                output = output[0]
            kh, kw = _pair(module.kernel_size)
            sh, sw = _pair(module.stride)
            dh, dw = _pair(getattr(module, "dilation", 1))
            ph, pw = _pair(module.padding) if op_type == "Conv2d" else (0, 0)
            geometries.append(LayerGeometry(
                op_type=op_type, name=name, stage=block_name,
                cin=x.shape[1], hin=x.shape[2], win=x.shape[3],
                cout=output.shape[1], hout=output.shape[2], wout=output.shape[3],
                kh=kh, kw=kw, sh=sh, sw=sw, dh=dh, dw=dw,
                groups=getattr(module, "groups", 1), ph=ph, pw=pw,
            ))
        return hook

    handles = []
    for name, module in model.named_modules():
        block_name = path_to_block.get(name)
        if block_name is None:
            continue
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook(name, block_name, "Conv2d")))
        elif isinstance(module, torch.nn.ConvTranspose2d):
            handles.append(module.register_forward_hook(make_hook(name, block_name, "ConvTranspose2d")))
        elif isinstance(module, torch.nn.MaxPool2d):
            handles.append(module.register_forward_hook(make_hook(name, block_name, "MaxPool2d")))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, in_channels, *input_hw))
    for h in handles:
        h.remove()
    return geometries, list(blocks.keys())


# ---- Extra hardware nodes with no conv/pool module (see finn_milp.md "Dataflow graph") ----

RESIDUAL_QUANT_BITS = 8  # QuantEltwiseAdd's input/output quant: Int8 regardless of bit_width= (Brevitas kwarg routing)
THRESHOLD_KINDS = ("skip_quant", "residual_add", "out_act", "input_quant", "act")
FIXED_BIT_KINDS = ("skip_quant", "residual_add")
PAD_MVAU_BITS = (2, 8)  # (weight, act) for the downsampling zero-pad MVAU -- PROVISIONAL, see finn_milp.md
# Real FINN v0.10.1 custom-op names (finn_milp.md "Dataflow graph" table) --
# every threshold-kind extra node and the pad-MVAU always use the RTL variant
# (extra_node_options hardcodes VARIANT_RTL_DSP_NOACT1 for both); the stream
# nodes (add/dup/concat/upsample) have no RTL alternative in this cost model.
EXTRA_OP_LABEL = {
    **{kind: "Thresholding_rtl" for kind in THRESHOLD_KINDS},
    "add": "AddStreams_hls", "dup": "DuplicateStreams_hls", "concat": "StreamingConcat_hls",
    "upsample": "UpsampleNearestNeighbour_hls", "pad_mvau": "MVAU_rtl",
}


@dataclass(frozen=True)
class ExtraNode:
    """A FINN node with no conv/pool module: residual-join/Initial thresholds,
    stream nodes, the downsampling pad-MVAU. fixed_bits None and non-empty
    bit_sources -> bits = max over bit_sources (the deployed rule)."""
    geom: LayerGeometry
    kind: str
    fixed_bits: int | None
    bit_sources: tuple[str, ...]


def trace_activation_shapes(model: torch.nn.Module, input_hw: tuple[int, int], in_channels: int) -> dict[str, tuple[int, int, int]]:
    """(C, H, W) of every `*.out_act` / `*.act` module output."""
    shapes: dict[str, tuple[int, int, int]] = {}
    handles = [
        module.register_forward_hook(lambda _m, _i, out, name=name: shapes.__setitem__(name, tuple(out.shape[1:])))
        for name, module in model.named_modules() if name.endswith((OUT_ACT_SUFFIX, ACT_SUFFIX))
    ]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, in_channels, *input_hw))
    for h in handles:
        h.remove()
    return shapes


def insert_skip_pads(
    dataflow_map: dict[str, list[str]], kinds: dict[str, str], geometries: list[LayerGeometry],
    act_shapes: dict[str, tuple[int, int, int]],
) -> None:
    """Downsampling skip: pool -> zero-pad concat, which FINN lowers to a 1x1 MVAU
    (cin -> cout) ahead of the skip requant. Inserted in place when channels grow."""
    geom = {g.name: g for g in geometries}
    for name, kind in list(kinds.items()):
        if kind != "skip_quant":
            continue
        block = name[: -len(SKIP_QUANT_SUFFIX)]
        pools = [p for p in dataflow_map[name] if p in geom and geom[p].op_type == "MaxPool2d"]
        if len(pools) != 1 or geom[pools[0]].cout == act_shapes[block + OUT_ACT_SUFFIX][0]:
            continue
        pad = block + SKIP_PAD_SUFFIX
        dataflow_map[pad] = pools
        kinds[pad] = "pad_mvau"
        dataflow_map[name] = [pad]


def _node_shape(
    name: str, dataflow_map: dict[str, list[str]], kinds: dict[str, str], geom: dict[str, LayerGeometry],
    act_shapes: dict[str, tuple[int, int, int]], input_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    """(C, H, W) of a node's output stream."""
    if name in geom:
        g = geom[name]
        return g.cout, g.hout, g.wout
    kind = kinds[name]
    if kind == "input_quant":
        return input_shape
    if kind == "dup":
        return _node_shape(dataflow_map[name][0], dataflow_map, kinds, geom, act_shapes, input_shape)
    if kind in ("act", "concat"):
        return act_shapes[name[: -len(CONCAT_SUFFIX)] + ACT_SUFFIX if kind == "concat" else name]
    if kind == "upsample":
        for consumer, preds in dataflow_map.items():
            if name in preds and consumer in geom:
                g = geom[consumer]
                return g.cin, g.hin, g.win
    block = name.rsplit(".", 1)[0]
    return act_shapes[block + OUT_ACT_SUFFIX]


def build_extra_nodes(
    dataflow_map: dict[str, list[str]], kinds: dict[str, str], act_shapes: dict[str, tuple[int, int, int]],
    geometries: list[LayerGeometry], predecessor_map: dict[str, list[str]], input_shape: tuple[int, int, int],
) -> list[ExtraNode]:
    """One ExtraNode per virtual node. Threshold bit sources are exactly what
    expand_layer_bits.py deploys; the pad-MVAU is a 1x1 conv geometry."""
    geom = {g.name: g for g in geometries}
    weight_names = {g.name: 0 for g in geometries if g.op_type != "MaxPool2d"}
    act_names = {g.name: 0 for g in geometries}
    def stage_of(n: str) -> str:
        return geom[n].stage if n in geom else n.rsplit(".", 1)[0]

    nodes = []
    for name, kind in kinds.items():
        channels, height, width = _node_shape(name, dataflow_map, kinds, geom, act_shapes, input_shape)
        if kind == "dup":  # belongs to the block consuming the fork -- pruning that block removes it
            consumer_stages = {stage_of(c) for c, preds in dataflow_map.items() if name in preds}
            stage = consumer_stages.pop() if len(consumer_stages) == 1 else stage_of(dataflow_map[name][0])
        else:
            stage = name.rsplit(".", 1)[0]
        if kind == "pad_mvau":
            pool = geom[dataflow_map[name][0]]
            node_geom = LayerGeometry(
                op_type="Conv2d", name=name, stage=stage, cin=pool.cout, hin=pool.hout, win=pool.wout,
                cout=channels, hout=pool.hout, wout=pool.wout, kh=1, kw=1, sh=1, sw=1,
            )
        else:
            node_geom = LayerGeometry(
                op_type=EXTRA_OP_LABEL[kind], name=name, stage=stage,
                cin=channels, hin=height, win=width, cout=channels, hout=height, wout=width, kh=1, kw=1, sh=1, sw=1,
            )
        if kind in FIXED_BIT_KINDS:
            nodes.append(ExtraNode(node_geom, kind, RESIDUAL_QUANT_BITS, ()))
        elif kind in THRESHOLD_KINDS:
            sources = tuple(s for s in resolve_act_sources(name, weight_names, act_names, predecessor_map) if s in act_names)
            if not sources:
                raise ValueError(f"{name}: no act-bit source resolvable (expand_layer_bits.resolve_act_sources).")
            nodes.append(ExtraNode(node_geom, kind, None, sources))
        else:
            nodes.append(ExtraNode(node_geom, kind, None, ()))
    return nodes


def extra_node_options(node: ExtraNode, force_dsp: bool) -> list[tuple[tuple, dict]]:
    """[(z key, cost dict)] for every legal (fold, bits) of an extra node -- the
    folding each FINN v0.10.1 op actually supports (see finn_milp.md)."""
    g, kind = node.geom, node.kind
    options = []
    if kind in THRESHOLD_KINDS:
        bit_options = (node.fixed_bits,) if node.fixed_bits is not None else CANDIDATE_BITS
        for pe in ([1] if FORCE_SERIAL else divisors(g.cout)):
            for ram_style in RAM_STYLES:
                for bits in bit_options:
                    key = (g.name, pe, 1, ram_style, VARIANT_RTL_DSP_NOACT1, 0, bits)
                    options.append((key, threshold_node_cost(g, bits, pe, ram_style=ram_style)))
    elif kind in FOLDABLE_STREAM_KINDS:
        for pe in ([1] if FORCE_SERIAL else divisors(g.cout)):
            options.append(((g.name, pe, 1, "none", "stream", 0, 0), stream_node_cost(kind, g, pe)))
    elif kind in STREAM_NODE_KINDS:
        options.append(((g.name, 1, 1, "none", "stream", 0, 0), stream_node_cost(kind, g)))
    elif kind == "pad_mvau":
        w, a = PAD_MVAU_BITS
        rtl_kwargs = _variant_cost_kwargs(VARIANT_RTL_DSP_NOACT1, force_dsp)
        for pe in ([1] if FORCE_SERIAL else divisors(max_pe(g))):
            for simd in ([1] if FORCE_SERIAL else divisors(max_simd(g))):
                cost = layer_cost_pe_simd(
                    g, w, a, pe, simd, RAM_STYLE_BLOCK, swu_ram_style="distributed",
                    **{**rtl_kwargs, "no_activation": False},
                )
                options.append(((g.name, pe, simd, "block", VARIANT_RTL_DSP_NOACT1, w, a), cost))
    else:
        raise ValueError(f"unknown extra node kind {kind!r}")
    return options


# ---- Search space and sensitivity helpers ----

def candidate_folds(layer: LayerGeometry) -> list[tuple[int, int, str, str]]:
    """Every (PE, SIMD, thr_ram_style, variant). MaxPool2d: one fixed sentinel fold."""
    if layer.op_type == "MaxPool2d":
        return [(1, 1, "block", VARIANT_RTL_DSP_NOACT1)]
    variants = [VARIANT_RTL_DSP_NOACT1]
    if ALLOW_LUT_MULT:
        variants.append(VARIANT_HLS_LUT_NOACT0)
    folds = [
        (pe, simd, ram_style, variant)
        for pe in divisors(max_pe(layer)) for simd in divisors(max_simd(layer))
        for ram_style in RAM_STYLES for variant in variants
    ]
    if FORCE_SERIAL:
        folds = [f for f in folds if f[0] == 1 and f[1] == 1]
    return folds


def _normalize(values: dict[tuple, float]) -> dict[tuple, float]:
    """Min-max scale to [0, 1] so sensitivity and cycles are comparable under alpha."""
    vals = values.values()
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span == 0:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / span for k, v in values.items()}


def _act_sensitivity_sources(name: str, predecessor_map: dict[str, list[str]] | None) -> list[str]:
    """A layer's act_bits is its INPUT stream, so read act sensitivity from its
    real predecessor(s); fall back to itself when there is none."""
    if predecessor_map is None:
        return [name]
    preds = predecessor_map.get(name)
    return preds if preds else [name]


def solve_joint_perlayer(
    sensitivity: dict, geometries: list[LayerGeometry], alpha: float,
    hard_lut_fraction: float, hard_bram_fraction: float,
    time_limit: int, gap_rel: float, max_cycles: float | None = None,
    pinned_bits: dict[str, tuple[int, int]] | None = None,
    predecessor_map: dict[str, list[str]] | None = None,
    force_dsp: bool = False,
    require_simd_ge_pe: bool = False,
    hard_dsp_fraction: float = 1.0,
    hard_uram_fraction: float = 1.0,
    optimize_downstream_rate: float | None = None,
    max_node_cycles: float | None = None,
    extra_nodes: list[ExtraNode] = (),
    dataflow_map: dict[str, list[str]] | None = None,
) -> dict:
    """Build and solve the MILP. Formulation: finn_milp.md "Formulation"."""
    candidate_pairs = tuple((w, a) for w in CANDIDATE_BITS for a in CANDIDATE_BITS)
    layer_names = tuple(g.name for g in geometries)
    n_layers = len(geometries)

    # ---- Variables: y[layer, w, a] -- one-hot bits per layer; sensitivity attaches here ----
    y = {
        (name, w, a): pulp.LpVariable(f"y_{name}_{w}_{a}", cat=pulp.LpBinary)
        for name in layer_names for w, a in candidate_pairs
    }
    raw_sensitivity: dict[tuple[str, int, int], float] = {}
    for name in layer_names:
        act_sources = [s for s in _act_sensitivity_sources(name, predecessor_map) if s in sensitivity]
        for w, a in candidate_pairs:
            sens_w = sensitivity[name]["sensitivity_w"][str(w)] if name in sensitivity else 0.0
            sens_a = max((sensitivity[s]["sensitivity_a"][str(a)] for s in act_sources), default=0.0)
            raw_sensitivity[(name, w, a)] = sens_w + sens_a
    sens_norm = _normalize(raw_sensitivity)

    # ---- Variables: z[layer, pe, simd, thr_ram_style, variant, w, a] -- one-hot fold x bits;
    # cycles / LUT / BRAM / DSP / URAM attach here ----
    z: dict[tuple, pulp.LpVariable] = {}
    layer_costs: dict[tuple, dict] = {}
    raw_cycles: dict[tuple, float] = {}
    raw_lut: dict[tuple, float] = {}
    raw_bram: dict[tuple, float] = {}
    raw_dsp: dict[tuple, float] = {}
    raw_uram: dict[tuple, float] = {}
    layer_folds: dict[str, list[tuple[int, int, str, str]]] = {}
    layer_cycle_terms: dict[str, list[tuple[pulp.LpVariable, float]]] = {g.name: [] for g in geometries}

    for layer in geometries:
        folds = candidate_folds(layer)
        if require_simd_ge_pe:
            folds = [f for f in folds if f[1] >= f[0]]
        layer_folds[layer.name] = folds
        for pe, simd, ram_style, variant in folds:
            variant_kwargs = _variant_cost_kwargs(variant, force_dsp)
            for w, a in candidate_pairs:
                # MVAU weights hard-fixed to block, SWU buffer to distributed (what real hardware does).
                cost = layer_cost_pe_simd(
                    layer, w, a, pe, simd, RAM_STYLE_BLOCK,
                    swu_ram_style="distributed", thr_ram_style=ram_style, **variant_kwargs,
                )
                key = (layer.name, pe, simd, ram_style, variant, w, a)
                layer_costs[key] = cost
                raw_cycles[key] = cost["cycles"]
                raw_lut[key] = calibrated_lut(
                    cost["total_lut"], w, a, force_dsp=_calibration_force_dsp(variant_kwargs),
                    lut_mult=(variant == VARIANT_HLS_LUT_NOACT0),
                )
                raw_bram[key] = calibrated_bram18k(
                    cost["swu_bram18"] + cost["wm_bram18"] + cost.get("thr_bram18", 0), w, a,
                    force_dsp=variant_kwargs["force_dsp"],
                )
                raw_dsp[key] = cost["total_dsp"]
                raw_uram[key] = cost.get("wm_uram18", 0) + cost.get("swu_uram18", 0) + cost.get("thr_uram18", 0)
                z[key] = pulp.LpVariable(f"z_{layer.name}_{pe}_{simd}_{ram_style}_{variant}_{w}_{a}", cat=pulp.LpBinary)
                layer_cycle_terms[layer.name].append((z[key], raw_cycles[key]))

    # ---- Variables: z for extra nodes -- key (name, pe, simd, ram_style, variant, w, bits) ----
    extra_keys: dict[str, list[tuple]] = {}
    rtl_kwargs = _variant_cost_kwargs(VARIANT_RTL_DSP_NOACT1, force_dsp)
    for node in extra_nodes:
        name = node.geom.name
        layer_cycle_terms[name] = []
        extra_keys[name] = []
        for key, cost in extra_node_options(node, force_dsp):
            _, pe, simd, ram_style, variant, w, bits = key
            calib_w = w or bits  # thresholds: (bits, bits) as for every standalone threshold
            layer_costs[key] = cost
            raw_cycles[key] = cost["cycles"]
            raw_lut[key] = calibrated_lut(cost["total_lut"], calib_w, bits, force_dsp=_calibration_force_dsp(rtl_kwargs))
            raw_bram[key] = calibrated_bram18k(
                cost["swu_bram18"] + cost["wm_bram18"] + cost["thr_bram18"], calib_w, bits, force_dsp=rtl_kwargs["force_dsp"],
            )
            raw_dsp[key] = cost["total_dsp"]
            raw_uram[key] = cost["wm_uram18"] + cost.get("swu_uram18", 0) + cost["thr_uram18"]
            z[key] = pulp.LpVariable(f"z_{name}_{pe}_{simd}_{ram_style}_{variant}_{w}_{bits}", cat=pulp.LpBinary)
            layer_cycle_terms[name].append((z[key], raw_cycles[key]))
            extra_keys[name].append(key)

    cycles_norm = _normalize(raw_cycles)
    layer_cycles_expr: dict[str, pulp.LpAffineExpression] = {
        name: pulp.lpSum(zvar * cyc for zvar, cyc in terms) for name, terms in layer_cycle_terms.items()
    }

    prob = pulp.LpProblem("FINN_MILP_perlayer", pulp.LpMinimize)

    # ---- Constraints ----

    # One (w, a) per layer; optional pin.
    for name in layer_names:
        prob += pulp.lpSum(y[(name, w, a)] for w, a in candidate_pairs) == 1, f"one_pair_per_layer_{name}"

    if pinned_bits is not None:
        missing = set(layer_names) - set(pinned_bits)
        if missing:
            raise ValueError(f"pinned_bits is missing layer(s): {sorted(missing)}")
        for name in layer_names:
            w_fixed, a_fixed = pinned_bits[name]
            if (w_fixed, a_fixed) not in candidate_pairs:
                raise ValueError(f"pinned_bits[{name!r}]=({w_fixed},{a_fixed}) not in candidate_pairs {candidate_pairs}")
            prob += y[(name, w_fixed, a_fixed)] == 1, f"pin_{name}"

    # Link: exactly one fold at the chosen (w, a).
    for layer in geometries:
        folds = layer_folds[layer.name]
        for w, a in candidate_pairs:
            prob += (
                pulp.lpSum(z[(layer.name, pe, simd, ram_style, variant, w, a)] for pe, simd, ram_style, variant in folds)
                == y[(layer.name, w, a)]
            ), f"link_{layer.name}_{w}_{a}"

    # Extra nodes: one option each; threshold bits with sources = max(source bits), exactly:
    # [bits_V >= b] = OR_s [bits_s >= b]  ->  >= each source, <= their sum.
    n_extra_constraints = 0
    for node in extra_nodes:
        name = node.geom.name
        prob += pulp.lpSum(z[k] for k in extra_keys[name]) == 1, f"one_option_{name}"
        n_extra_constraints += 1
        if not node.bit_sources:
            continue
        for level in sorted(CANDIDATE_BITS)[1:]:
            node_at_least = pulp.lpSum(z[k] for k in extra_keys[name] if k[-1] >= level)
            sources_at_least = [
                pulp.lpSum(y[(s, w, a)] for w, a in candidate_pairs if a >= level) for s in node.bit_sources
            ]
            for i, source_expr in enumerate(sources_at_least):
                prob += node_at_least >= source_expr, f"thr_bits_ge_{name}_{level}_{i}"
            prob += node_at_least <= pulp.lpSum(sources_at_least), f"thr_bits_le_{name}_{level}"
            n_extra_constraints += len(sources_at_least) + 1

    # Rate coherence (--optimize-downstream-rate), on the dataflow graph.
    topology = dataflow_map if dataflow_map is not None else predecessor_map
    fixed_cycle_names = {name for name, terms in layer_cycle_terms.items() if len({c for _, c in terms}) == 1}
    n_join_constraints = 0
    n_chain_rate_constraints = 0
    if optimize_downstream_rate is not None:
        if topology is None:
            raise ValueError("optimize_downstream_rate needs a predecessor_map or dataflow_map.")

        # Join balance: sibling branches within ratio, both ways (fixed-cycle nodes -- pools,
        # concat, upsample -- exempt: nothing to fold on that side).
        unique_joins = {
            tuple(sorted(preds)) for preds in topology.values()
            if len(preds) >= 2 and all(p in layer_cycles_expr for p in preds)
        }
        for branches in unique_joins:
            foldable_branches = [b for b in branches if b not in fixed_cycle_names]
            for branch_i in foldable_branches:
                for branch_j in foldable_branches:
                    if branch_i == branch_j:
                        continue
                    prob += (
                        layer_cycles_expr[branch_i] <= optimize_downstream_rate * layer_cycles_expr[branch_j]
                    ), f"join_balance_{branch_i}_vs_{branch_j}"
                    n_join_constraints += 1

        # Chain: rate[L] <= ratio * rate[D] for every descendant D, in O(edges) via
        # max_downstream_rate (the SLOWEST/bottleneck descendant rate) bounded BELOW (so it
        # can't be deflated). rate = cycles / outputs. Constraining L against the slowest thing
        # downstream is what actually catches a fast upstream node outrunning a distant slow
        # bottleneck (the compounding-mismatch case) -- an earlier version tracked the FASTEST
        # descendant (min, bounded above), which only restricted the opposite, harmless
        # direction (a slow producer merely starves a FIFO, no depth risk) and left the
        # documented target case ("producer outruns consumer") completely unconstrained;
        # confirmed by a toy 3-node chain (see finn_milp.md "Rate coherence"). Same
        # anti-inflation principle as before, mirrored: a MAX-type auxiliary appearing on the
        # restricted (left) side of a <= must be bounded from below only, so it can't be pushed
        # down below the true maximum to cheat the constraint.
        node_shapes = [*geometries, *(node.geom for node in extra_nodes)]
        rate_expr = {
            g.name: layer_cycles_expr[g.name] * (1.0 / (g.hout * g.wout * g.cout))
            for g in node_shapes if g.name in layer_cycles_expr
        }
        successors: dict[str, list[str]] = {name: [] for name in rate_expr}
        for consumer, preds in topology.items():
            if consumer not in rate_expr:
                continue
            for pred in preds:
                if pred in successors:
                    successors[pred].append(consumer)
        max_downstream_rate = {
            name: pulp.LpVariable(f"max_downstream_rate_{name}", lowBound=0)
            for name, children in successors.items() if children
        }
        # Fixed-cycle nodes (MaxPool, concat, upsample -- one cycle value whatever the
        # option, same set the join-balance check exempts) have no folding freedom, so
        # pairing them against a foldable node forces all the give onto the foldable
        # side. Excluded from contributing their OWN rate to an ancestor's
        # max_downstream_rate, and exempted from their own outer constraint -- but still
        # relayed transparently (their max_downstream_rate[child] link stays), so a real
        # foldable bottleneck sitting beyond a fixed pass-through node is still caught.
        for name, children in successors.items():
            if not children:
                continue
            for child in children:
                if child not in fixed_cycle_names:
                    prob += max_downstream_rate[name] >= rate_expr[child], f"max_rate_{name}_ge_{child}"
                    n_chain_rate_constraints += 1
                if child in max_downstream_rate:
                    prob += max_downstream_rate[name] >= max_downstream_rate[child], f"max_rate_{name}_ge_max_{child}"
                    n_chain_rate_constraints += 1
            if name in fixed_cycle_names:
                continue
            prob += (
                max_downstream_rate[name] <= optimize_downstream_rate * rate_expr[name]
            ), f"downstream_rate_{name}"
            n_chain_rate_constraints += 1

    # Hard resource budgets and latency cap (sum of cycles).
    prob += pulp.lpSum(z[k] * raw_lut[k] for k in z) <= hard_lut_fraction * XCZU7EV["LUT"], "hard_lut_budget"
    prob += pulp.lpSum(z[k] * raw_bram[k] for k in z) <= hard_bram_fraction * XCZU7EV["BRAM_18K"], "hard_bram_budget"
    prob += pulp.lpSum(z[k] * raw_dsp[k] for k in z) <= hard_dsp_fraction * XCZU7EV["DSP"], "hard_dsp_budget"
    prob += pulp.lpSum(z[k] * raw_uram[k] for k in z) <= hard_uram_fraction * XCZU7EV["URAM"], "hard_uram_budget"
    if max_cycles is not None:
        prob += pulp.lpSum(z[k] * raw_cycles[k] for k in z) <= max_cycles, "max_cycles_budget"

    # Throughput (--target-fps): a pipeline delivers one frame per slowest-node frame time,
    # so EVERY node must fit in max_node_cycles. Nodes that can't even fully folded are reported.
    min_node_cycles = {name: min(c for _, c in terms) for name, terms in layer_cycle_terms.items()}
    throughput_floor_violations = {}
    if max_node_cycles is not None:
        throughput_floor_violations = {n: c for n, c in min_node_cycles.items() if c > max_node_cycles}
        for name, expr in layer_cycles_expr.items():
            prob += expr <= max_node_cycles, f"throughput_{name}"

    # ---- Objective: alpha * mean(sens_norm over layers) + (1 - alpha) * mean(cycles_norm over hardware nodes) ----
    sensitivity_term = (1.0 / n_layers) * pulp.lpSum(
        y[(name, w, a)] * sens_norm[(name, w, a)] for name in layer_names for w, a in candidate_pairs
    )
    n_hardware_nodes = n_layers + len(extra_nodes)
    latency_term = (1.0 / n_hardware_nodes) * pulp.lpSum(z[k] * cycles_norm[k] for k in z)
    prob += alpha * sensitivity_term + (1 - alpha) * latency_term

    # ---- Solve and extract ----
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap_rel))
    status_name = pulp.LpStatus[status]
    if status_name not in ("Optimal", "Infeasible"):
        raise RuntimeError(f"per-layer ILP hit an unexpected solver status: {status_name!r}.")

    n_binary_vars = len(y) + len(z)
    n_constraints = (
        n_layers + sum(len(candidate_pairs) for _ in geometries) + 4
        + (1 if max_cycles is not None else 0) + n_join_constraints + n_chain_rate_constraints
        + n_extra_constraints
    )

    if status_name != "Optimal":
        return {
            "status": status_name,
            "alpha": alpha,
            "layer_weight_bits": {}, "layer_act_bits": {}, "per_layer": {}, "extra_nodes": {},
            "_diagnostics": {
                "alpha": alpha, "candidate_bits": list(CANDIDATE_BITS), "n_layers": n_layers,
                "n_extra_nodes": len(extra_nodes),
                "n_binary_vars": n_binary_vars, "n_constraints": n_constraints,
                "hard_lut_fraction": hard_lut_fraction, "hard_bram_fraction": hard_bram_fraction,
                "hard_dsp_fraction": hard_dsp_fraction, "hard_uram_fraction": hard_uram_fraction,
                "max_cycles": max_cycles, "max_node_cycles": max_node_cycles,
                "throughput_floor_violations": throughput_floor_violations,
                "optimize_downstream_rate": optimize_downstream_rate, "n_join_constraints": n_join_constraints,
                "n_chain_rate_constraints": n_chain_rate_constraints,
                "solver_time_limit_s": time_limit, "solver_gap_rel": gap_rel, "force_serial": FORCE_SERIAL,
                "force_dsp": force_dsp, "require_simd_ge_pe": require_simd_ge_pe, "allow_lut_mult": ALLOW_LUT_MULT,
                "note": f"Solver status {status_name!r} -- no joint per-layer (bits, folding) assignment "
                        f"satisfies the requested hard LUT/BRAM/DSP/URAM budget(s) (hard_lut_fraction={hard_lut_fraction}, "
                        f"hard_bram_fraction={hard_bram_fraction}, hard_dsp_fraction={hard_dsp_fraction}, "
                        f"hard_uram_fraction={hard_uram_fraction})"
                        + (f" and max_cycles={max_cycles:.0f}" if max_cycles is not None else "")
                        + (f" and max_node_cycles={max_node_cycles:.0f} (nodes that cannot reach it even fully "
                           f"folded: {throughput_floor_violations or 'none'})" if max_node_cycles is not None else "")
                        + (f" and optimize_downstream_rate={optimize_downstream_rate}" if optimize_downstream_rate is not None else "")
                        + " at all.",
            },
        }

    layer_weight_bits: dict[str, int] = {}
    layer_act_bits: dict[str, int] = {}
    for name in layer_names:
        chosen = [(w, a) for w, a in candidate_pairs if pulp.value(y[(name, w, a)]) > 0.5]
        assert len(chosen) == 1, f"layer {name}: expected exactly one (w,a) pair chosen, got {chosen}"
        layer_weight_bits[name], layer_act_bits[name] = chosen[0]

    per_layer: dict[str, dict] = {}
    for layer in geometries:
        w, a = layer_weight_bits[layer.name], layer_act_bits[layer.name]
        folds = layer_folds[layer.name]
        chosen_fold = None
        for pe, simd, ram_style, variant in folds:
            if pulp.value(z[(layer.name, pe, simd, ram_style, variant, w, a)]) > 0.5:
                chosen_fold = (pe, simd, ram_style, variant)
                break
        assert chosen_fold is not None, f"layer {layer.name}: no folding choice selected at its own chosen (w,a)={w,a}"
        pe, simd, ram_style, variant = chosen_fold
        cost = layer_costs[(layer.name, pe, simd, ram_style, variant, w, a)]
        variant_kwargs = _variant_cost_kwargs(variant, force_dsp)
        per_layer[layer.name] = {
            "stage": layer.stage, "pe": pe, "simd": simd, "thr_ram_style": ram_style, "variant": variant,
            "force_dsp": variant_kwargs["force_dsp"], "mvau_noAct": variant_kwargs["no_activation"],
            "weight_bits": w, "act_bits": a, **cost,
        }

    for v in per_layer.values():
        v["lut_calibrated"] = calibrated_lut(
            v["total_lut"], v["weight_bits"], v["act_bits"],
            force_dsp=_calibration_force_dsp(_variant_cost_kwargs(v["variant"], force_dsp)),
            lut_mult=(v["variant"] == VARIANT_HLS_LUT_NOACT0),
        )
        v["bram18k_calibrated"] = calibrated_bram18k(
            v["swu_bram18"] + v["wm_bram18"] + v.get("thr_bram18", 0), v["weight_bits"], v["act_bits"],
            force_dsp=_variant_cost_kwargs(v["variant"], force_dsp)["force_dsp"],
        )
    total_lut = sum(v["lut_calibrated"] for v in per_layer.values())
    total_bram = sum(v["bram18k_calibrated"] for v in per_layer.values())
    total_uram = sum(
        v.get("wm_uram18", 0) + v.get("swu_uram18", 0) + v.get("thr_uram18", 0) for v in per_layer.values()
    )
    total_dsp = sum(v["total_dsp"] for v in per_layer.values())
    total_cycles = sum(v["cycles"] for v in per_layer.values())

    extra_out: dict[str, dict] = {}
    for node in extra_nodes:
        name = node.geom.name
        chosen = next(k for k in extra_keys[name] if pulp.value(z[k]) > 0.5)
        _, pe, simd, ram_style, _, w, bits = chosen
        bits_rule = "fixed" if node.fixed_bits is not None else ("max_of_sources" if node.bit_sources else "n/a")
        extra_out[name] = {
            "kind": node.kind, "stage": node.geom.stage, "pe": pe, "simd": simd, "ram_style": ram_style,
            "weight_bits": w or None, "act_bits": bits or None, "bits_rule": bits_rule,
            "bit_sources": list(node.bit_sources), "channels": node.geom.cout,
            "cycles": raw_cycles[chosen], "lut_calibrated": raw_lut[chosen],
            "bram18k_calibrated": raw_bram[chosen], "dsp": raw_dsp[chosen], "uram18": raw_uram[chosen],
        }
    extra_by_kind: dict[str, dict] = {}
    for v in extra_out.values():
        agg = extra_by_kind.setdefault(v["kind"], {"n": 0, "lut_calibrated": 0.0, "bram18k_calibrated": 0.0, "dsp": 0, "cycles": 0})
        agg["n"] += 1
        for field in ("lut_calibrated", "bram18k_calibrated", "dsp", "cycles"):
            agg[field] += v[field]
    extra_lut = sum(v["lut_calibrated"] for v in extra_out.values())
    extra_bram = sum(v["bram18k_calibrated"] for v in extra_out.values())
    extra_cycles = sum(v["cycles"] for v in extra_out.values())
    total_lut += extra_lut
    total_bram += extra_bram
    total_uram += sum(v["uram18"] for v in extra_out.values())
    total_dsp += sum(v["dsp"] for v in extra_out.values())
    total_cycles += extra_cycles
    node_cycles = {**{n: v["cycles"] for n, v in per_layer.items()}, **{n: v["cycles"] for n, v in extra_out.items()}}
    bottleneck_node = max(node_cycles, key=node_cycles.get)

    return {
        "status": status_name,
        "alpha": alpha,
        "layer_weight_bits": layer_weight_bits,
        "layer_act_bits": layer_act_bits,
        "per_layer": per_layer,
        "extra_nodes": extra_out,
        "_diagnostics": {
            "alpha": alpha, "candidate_bits": list(CANDIDATE_BITS), "n_layers": n_layers,
            "n_binary_vars": n_binary_vars, "n_constraints": n_constraints,
            "n_extra_nodes": len(extra_nodes), "extra_lut_calibrated": extra_lut,
            "extra_bram18k_calibrated": extra_bram, "extra_cycles": extra_cycles, "extra_by_kind": extra_by_kind,
            "total_lut_calibrated": total_lut, "xczu7ev_lut_budget": XCZU7EV["LUT"],
            "lut_pct_of_budget": 100 * total_lut / XCZU7EV["LUT"],
            "total_bram18k_calibrated": total_bram, "xczu7ev_bram18k_budget": XCZU7EV["BRAM_18K"],
            "bram_pct_of_budget": 100 * total_bram / XCZU7EV["BRAM_18K"],
            "total_uram18": total_uram, "xczu7ev_uram_budget": XCZU7EV["URAM"],
            "uram_pct_of_budget": 100 * total_uram / XCZU7EV["URAM"], "total_cycles": total_cycles,
            "total_dsp": total_dsp, "xczu7ev_dsp_budget": XCZU7EV["DSP"],
            "dsp_pct_of_budget": 100 * total_dsp / XCZU7EV["DSP"],
            "hard_lut_fraction": hard_lut_fraction, "hard_bram_fraction": hard_bram_fraction,
            "hard_dsp_fraction": hard_dsp_fraction, "hard_uram_fraction": hard_uram_fraction,
            "max_cycles": max_cycles, "max_node_cycles": max_node_cycles,
            "bottleneck_node": bottleneck_node, "bottleneck_cycles": node_cycles[bottleneck_node],
            "optimize_downstream_rate": optimize_downstream_rate, "n_join_constraints": n_join_constraints,
            "n_chain_rate_constraints": n_chain_rate_constraints,
            "solver_time_limit_s": time_limit, "solver_gap_rel": gap_rel, "force_serial": FORCE_SERIAL,
            "force_dsp": force_dsp, "require_simd_ge_pe": require_simd_ge_pe, "allow_lut_mult": ALLOW_LUT_MULT,
            "note": "Joint per-LAYER MILP: y[layer,w,a] (sensitivity) linked to z[layer,pe,simd,ram_style,w,a] "
                    "(cycles/LUT/BRAM/DSP/URAM) via a same-layer equality constraint -- LUT/BRAM/DSP/URAM are REAL "
                    "hard <= XCZU7EV constraints here (not a soft penalty). ram_style selects the SWU's own line-"
                    "buffer memory (2026-09-17: MVAU's own weight-memory ram_style is hard-fixed to block, ultra "
                    "disabled -- real hardware never used it, see candidate_folds' own docstring). Objective = "
                    "alpha*mean(sens_norm) + (1-alpha)*mean(cycles_norm), both mean-normalized by the SAME "
                    "n_layers. GUARANTEED to fit the requested hard budget(s) under this cost model's own "
                    "calibration -- a steering signal at that calibration, not a certified hardware guarantee. "
                    "Coarser than nnunetv2.nets.LayerQuantENet's own full per-quantizer-site deployment schema -- "
                    "see module docstring's SCOPE BOUNDARY.",
        },
    }


# ---- Sweep summary (summary.csv + run_args.json, one row per alpha) ----

_SUMMARY_FIELDS = [
    "alpha", "status", "avg_weight_bits", "avg_act_bits",
    "lut_pct_of_budget", "bram_pct_of_budget", "dsp_pct_of_budget",
    "total_dsp", "total_cycles", "clock_mhz", "latency_ms", "target_fps", "fps", "bottleneck_node",
    "n_binary_vars", "n_layers", "n_zero_sensitivity_layers", "zero_sensitivity_layers",
]


def _update_sweep_summary(
    out_dir: Path, args: argparse.Namespace, result: dict, zero_sensitivity_layers: list[str],
) -> None:
    """Upsert this alpha's row; warn if shared args differ from earlier alphas."""
    diag = result["_diagnostics"]
    weight_bits, act_bits = result.get("layer_weight_bits", {}), result.get("layer_act_bits", {})
    avg_weight_bits = sum(weight_bits.values()) / len(weight_bits) if weight_bits else float("nan")
    avg_act_bits = sum(act_bits.values()) / len(act_bits) if act_bits else float("nan")
    total_cycles = diag.get("total_cycles")
    latency_ms = total_cycles / (args.clock_mhz * 1000) if total_cycles is not None else float("nan")

    row = {
        "alpha": args.alpha, "status": result["status"],
        "avg_weight_bits": avg_weight_bits, "avg_act_bits": avg_act_bits,
        "lut_pct_of_budget": diag.get("lut_pct_of_budget"), "bram_pct_of_budget": diag.get("bram_pct_of_budget"),
        "dsp_pct_of_budget": diag.get("dsp_pct_of_budget"), "total_dsp": diag.get("total_dsp"),
        "total_cycles": total_cycles, "clock_mhz": args.clock_mhz, "latency_ms": latency_ms,
        "target_fps": args.target_fps, "bottleneck_node": diag.get("bottleneck_node"),
        "fps": args.clock_mhz * 1e6 / diag["bottleneck_cycles"] if diag.get("bottleneck_cycles") else float("nan"),
        "n_binary_vars": diag.get("n_binary_vars"), "n_layers": diag.get("n_layers"),
        "n_zero_sensitivity_layers": len(zero_sensitivity_layers),
        "zero_sensitivity_layers": ";".join(zero_sensitivity_layers),
    }

    summary_path = out_dir / "summary.csv"
    rows = []
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if float(r["alpha"]) != args.alpha]
    rows.append(row)
    rows.sort(key=lambda r: float(r["alpha"]))
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {summary_path} ({len(rows)} alpha rows).")

    run_args_path = out_dir / "run_args.json"
    shared_args = {
        "config": args.config, "sensitivity-file": str(args.sensitivity_file),
        "candidate-bits": ",".join(str(b) for b in CANDIDATE_BITS),
        "hard-lut-fraction": args.hard_lut_fraction, "hard-bram-fraction": args.hard_bram_fraction,
        "hard-dsp-fraction": args.hard_dsp_fraction, "hard-uram-fraction": args.hard_uram_fraction, "force-dsp": args.force_dsp,
        "max-latency-ms": args.max_latency_ms, "clock-mhz": args.clock_mhz, "target-fps": args.target_fps,
        "optimize-downstream-rate": args.optimize_downstream_rate,
        "force-serial": FORCE_SERIAL, "require-simd-ge-pe": args.require_simd_ge_pe,
        "allow-lut-mult": ALLOW_LUT_MULT,
        "time-limit": args.time_limit, "gap-rel": args.gap_rel,
    }
    existing_alphas = []
    if run_args_path.exists():
        existing = json.loads(run_args_path.read_text())
        existing_alphas = existing.get("alphas", [])
        prior_shared = existing.get("shared_args", {})
        mismatched = {k: (prior_shared[k], v) for k, v in shared_args.items() if k in prior_shared and prior_shared[k] != v}
        if mismatched:
            print(f"WARNING: {run_args_path} was last written with DIFFERENT shared args for other alpha(s) in "
                  f"this sweep: {mismatched} -- overwriting shared_args with THIS run's values; the sweep may "
                  f"now be inconsistent across alphas (mixed hard caps/config/etc).")
    run_args = {
        "pipeline": [
            "MILP/layer_sensitivity.py --candidate-bits ...",
            "MILP/finn_milp.py --candidate-bits ... --hard-lut-fraction ... --hard-bram-fraction ... "
            "--hard-dsp-fraction ... --force-dsp",
            "MILP/expand_layer_bits.py",
        ],
        "shared_args": shared_args,
        "alphas": sorted(set(existing_alphas) | {args.alpha}),
        "granularity": "layer",
        "notes": f"Per-layer joint bits+folding MILP for {args.config}, hard LUT/BRAM/DSP caps "
                 f"{args.hard_lut_fraction}/{args.hard_bram_fraction}/{args.hard_dsp_fraction} under "
                 f"force_dsp={args.force_dsp}. See summary.csv for per-alpha results.",
    }
    run_args_path.write_text(json.dumps(run_args, indent=2))
    print(f"Updated {run_args_path}.")


# ---- CLI (flag rationale: finn_milp.md "CLI flags") ----

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="MILP/configs/config_*.py module to load.")
    parser.add_argument("--sensitivity-file", type=Path, required=True,
                         help="layer_sensitivity_*.json from layer_sensitivity.py.")
    parser.add_argument("--candidate-bits", type=str, default=None,
                         help="Comma-separated bit-width candidates (e.g. '4,6,8'); overrides CANDIDATE_BITS.")
    parser.add_argument("--alpha", type=float, required=True,
                         help="1.0 = sensitivity only, 0.0 = cycles only.")
    parser.add_argument("--hard-lut-fraction", type=float, default=1.0, help="Hard cap as a fraction of device LUT.")
    parser.add_argument("--hard-bram-fraction", type=float, default=1.0, help="Hard cap as a fraction of device BRAM_18K.")
    parser.add_argument("--hard-dsp-fraction", type=float, default=1.0, help="Hard cap as a fraction of device DSP.")
    parser.add_argument("--hard-uram-fraction", type=float, default=1.0,
                         help="Hard cap as a fraction of device URAM (inert: URAM is always 0).")
    parser.add_argument("--force-dsp", action="store_true",
                         help="Use the forced-DSP calibration factors instead of the auto-resType table.")
    parser.add_argument("--target-fps", type=float, default=None,
                         help="Hard throughput target: every node's cycles per frame <= clock-mhz*1e6/target-fps.")
    parser.add_argument("--max-latency-ms", type=float, default=None,
                         help="Hard cap on the sum of cycles, as ms at --clock-mhz.")
    parser.add_argument("--clock-mhz", type=float, default=100.0, help="Clock for --max-latency-ms (default 100).")
    parser.add_argument("--optimize-downstream-rate", type=float, default=None,
                         help="Ratio for join-balance + downstream-rate coherence constraints (off by default).")
    parser.add_argument("--force-serial", action="store_true", help="Restrict every node to PE=SIMD=1.")
    parser.add_argument("--allow-lut-mult", action="store_true",
                         help="Also allow the hls_lut_noact0 variant (LUT multipliers, fused activation).")
    parser.add_argument("--require-simd-ge-pe", action="store_true", help="Drop conv folds with PE > SIMD.")
    parser.add_argument("--time-limit", type=int, default=1800, help="CBC time limit in seconds.")
    parser.add_argument("--gap-rel", type=float, default=0.02, help="CBC relative optimality gap.")
    parser.add_argument("--pin-bits-file", type=Path, default=None,
                         help="TEST-ONLY: pin y to a layer_bits_*.json (alpha then has no effect).")
    parser.add_argument("--out-file", type=Path, required=True)
    args = parser.parse_args()

    load_config(args.config)

    if args.candidate_bits is not None:
        global CANDIDATE_BITS
        CANDIDATE_BITS = tuple(sorted(int(b) for b in args.candidate_bits.split(",")))
        print(f"Overriding CANDIDATE_BITS to {CANDIDATE_BITS} (from --candidate-bits).")

    if args.force_serial:
        global FORCE_SERIAL
        FORCE_SERIAL = True
        print("--force-serial: every layer restricted to (PE, SIMD) = (1, 1) before solving.")

    if args.allow_lut_mult:
        global ALLOW_LUT_MULT
        ALLOW_LUT_MULT = True
        print("--allow-lut-mult: 'hls_lut_noact0' (LUT-mult, fused activation) made eligible alongside "
              "'rtl_dsp_noact1' on every layer -- PROVISIONAL fused-threshold LUT term, see --help.")

    with open(args.sensitivity_file) as f:
        sensitivity = json.load(f)
    if "_zero_sensitivity_layers" in sensitivity:
        zero_sensitivity_layers = sensitivity.pop("_zero_sensitivity_layers")
    else:
        zero_sensitivity_layers = [
            name for name, entry in sensitivity.items()
            if abs(entry.get("trace_w", 1.0)) < 1e-9 and abs(entry.get("trace_a", 1.0)) < 1e-9
        ]
    if zero_sensitivity_layers:
        print(f"WARNING: {len(zero_sensitivity_layers)} layer(s) in {args.sensitivity_file} have ZERO measured "
              f"sensitivity (see layer_sensitivity.py's own detector) -- their bit-width choice cannot affect "
              f"accuracy at all, which usually means the layer is dead (e.g. a collapsed BatchNorm permanently "
              f"off its own ReLU). The ILP will still cheaply-quantize them (correctly, given zero sensitivity), "
              f"but a dead layer's resulting near-zero calibrated quantizer scale can silently underflow to "
              f"exactly 0.0 under fp16 deployment -- consider pruning them instead (ENet.py's "
              f"apply_block_pruning / ENET_PRUNED_BLOCKS): {zero_sensitivity_layers}")

    model = ENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, decoder_type=DECODER_TYPE,
        use_asymmetric=USE_ASYMMETRIC, context_pattern=CONTEXT_PATTERN,
        separable_dilated=SEPARABLE_DILATED, use_prelu=globals().get("USE_PRELU", True), prelu_variant=PRELU_VARIANT,
        use_dsc=globals().get("USE_DSC", False), dsc_no_projection=globals().get("DSC_NO_PROJECTION", False),
        dsc_no_projection_context_only=globals().get("DSC_NO_PROJECTION_CONTEXT_ONLY", False),
        reg_bookend_dsc=globals().get("REG_BOOKEND_DSC", False),
        dsc_separable=globals().get("DSC_SEPARABLE", False),
    )
    geometries, _block_names = trace_layer_geometry(model, INPUT_HW, IN_CHANNELS)
    layer_names = tuple(g.name for g in geometries)

    # Conv-only predecessor map: drives act-sensitivity sources and deployed bits.
    # Non-fatal -- falls back to self-indexed act sensitivity.
    try:
        predecessor_map = compute_predecessor_map(model)
        n_resolved = sum(1 for name in layer_names if predecessor_map.get(name))
        print(f"Predecessor map: {n_resolved}/{len(layer_names)} layers have a real, traced predecessor "
              f"(the rest are the network's own first layer(s), with no real predecessor to fall back to "
              f"anything but self-sensitivity).")
    except Exception as error:
        predecessor_map = None
        print(f"WARNING: could not compute a predecessor map ({type(error).__name__}: {error}) -- falling back "
              f"to self-indexed act sensitivity for EVERY layer.")

    # Real FINN dataflow graph incl. every node with no conv/pool module. Fatal on
    # failure: skipping them would silently under-price every solve.
    dataflow_map, node_kinds = compute_dataflow_graph(model)
    act_shapes = trace_activation_shapes(model, INPUT_HW, IN_CHANNELS)
    insert_skip_pads(dataflow_map, node_kinds, geometries, act_shapes)
    extra_nodes = build_extra_nodes(
        dataflow_map, node_kinds, act_shapes, geometries, predecessor_map or {}, (IN_CHANNELS, *INPUT_HW),
    )
    kind_counts = {kind: sum(n.kind == kind for n in extra_nodes) for kind in dict.fromkeys(n.kind for n in extra_nodes)}
    print(f"Extra dataflow nodes: {len(extra_nodes)} {kind_counts}.")

    maxpool_names = {g.name for g in geometries if g.op_type == "MaxPool2d"}
    sensitivity_names = set(sensitivity.keys())
    missing_in_sensitivity = set(layer_names) - sensitivity_names
    unexpected_missing = missing_in_sensitivity - maxpool_names
    if unexpected_missing:
        raise ValueError(
            f"--sensitivity-file is missing entries for: {sorted(unexpected_missing)} -- traced model layers "
            f"and --sensitivity-file's own top-level keys must match 1:1 for every Conv2d/ConvTranspose2d layer "
            f"(only MaxPool2d layers are allowed to be absent, see module docstring)."
        )
    if missing_in_sensitivity:
        print(f"Note: {sorted(missing_in_sensitivity)} have no sensitivity entry (MaxPool2d, no weight, never "
              f"HAWQ-measured) -- given a fixed raw sensitivity of 0.0 for every (w,a), see module docstring.")

    candidate_pairs_count = len(CANDIDATE_BITS) ** 2
    n_folds_per_layer = [
        len([f for f in candidate_folds(g) if f[1] >= f[0]]) if args.require_simd_ge_pe
        else len(candidate_folds(g))
        for g in geometries
    ]
    n_z = sum(n_folds_per_layer) * candidate_pairs_count
    n_y = len(layer_names) * candidate_pairs_count
    print(f"Traced {len(geometries)} layers (per-layer granularity, no block grouping). "
          f"candidate_bits={CANDIDATE_BITS} -> {candidate_pairs_count} (w,a) pairs. "
          f"y: {n_y} binaries, z: {n_z} binaries ({n_y + n_z} total). Solving (time_limit={args.time_limit}s, "
          f"gap_rel={args.gap_rel})...")

    pinned_bits = None
    if args.pin_bits_file is not None:
        with open(args.pin_bits_file) as f:
            pin_source = json.load(f)
        pinned_bits = {
            name: (pin_source["layer_weight_bits"][name], pin_source["layer_act_bits"][name]) for name in layer_names
        }
        print(f"--pin-bits-file: y pinned to {args.pin_bits_file} for all {len(layer_names)} layers (alpha ignored).")

    max_node_cycles = None
    if args.target_fps is not None:
        max_node_cycles = args.clock_mhz * 1e6 / args.target_fps
        print(f"--target-fps {args.target_fps} @ {args.clock_mhz}MHz -> every node <= {max_node_cycles:.0f} cycles "
              f"(hard constraint).")

    max_cycles = None
    if args.max_latency_ms is not None:
        max_cycles = args.max_latency_ms / 1000 * args.clock_mhz * 1e6
        print(f"--max-latency-ms {args.max_latency_ms} @ {args.clock_mhz}MHz -> max_cycles={max_cycles:.0f} "
              f"(hard constraint).")

    result = solve_joint_perlayer(
        sensitivity, geometries, args.alpha, args.hard_lut_fraction, args.hard_bram_fraction,
        args.time_limit, args.gap_rel, max_cycles=max_cycles, pinned_bits=pinned_bits,
        predecessor_map=predecessor_map, force_dsp=args.force_dsp, require_simd_ge_pe=args.require_simd_ge_pe,
        hard_dsp_fraction=args.hard_dsp_fraction, hard_uram_fraction=args.hard_uram_fraction,
        optimize_downstream_rate=args.optimize_downstream_rate, max_node_cycles=max_node_cycles,
        extra_nodes=extra_nodes, dataflow_map=dataflow_map,
    )
    hardware_nodes = [*geometries, *(node.geom for node in extra_nodes)]
    result["dataflow_graph"] = {
        "input": [IN_CHANNELS, *INPUT_HW],
        "edges": dataflow_map,
        "shapes": {g.name: [g.cout, g.hout, g.wout] for g in hardware_nodes},
        "op_types": {**{g.name: g.op_type for g in geometries}, **{n.geom.name: EXTRA_OP_LABEL[n.kind] for n in extra_nodes}},
        "kinds": node_kinds,
    }

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out_file}")
    print(f"ILP status: {result['status']}")
    if result["status"] != "Optimal":
        print(f"No joint per-layer (bits, folding) assignment satisfies the requested hard budget(s) -- see "
              f"{args.out_file}'s own _diagnostics.note.")
        return
    diag = result["_diagnostics"]
    print(f"LUT used (calibrated): {diag['total_lut_calibrated']:.0f} ({diag['lut_pct_of_budget']:.1f}% of "
          f"{XCZU7EV['LUT']} budget) -- GUARANTEED (hard constraint).")
    print(f"BRAM_18K used (calibrated): {diag['total_bram18k_calibrated']:.0f} ({diag['bram_pct_of_budget']:.1f}% "
          f"of {XCZU7EV['BRAM_18K']} budget) -- GUARANTEED (hard constraint).")
    print(f"DSP used: {diag['total_dsp']:.0f} ({diag['dsp_pct_of_budget']:.1f}% of {XCZU7EV['DSP']} budget) "
          f"-- GUARANTEED (hard constraint).")
    print(f"Bottleneck: {diag['bottleneck_node']} at {diag['bottleneck_cycles']:.0f} cycles/frame -> "
          f"{args.clock_mhz * 1e6 / diag['bottleneck_cycles']:.1f} FPS @ {args.clock_mhz}MHz")
    print(f"Total cycles (sum over nodes): {diag['total_cycles']:.0f}")
    for kind, agg in diag["extra_by_kind"].items():
        print(f"  {kind:13s} x{agg['n']:3d}: LUT {agg['lut_calibrated']:8.0f}  BRAM_18K {agg['bram18k_calibrated']:6.1f}  "
              f"DSP {agg['dsp']:4.0f}  cycles {agg['cycles']:10.0f}")

    for path in write_outputs(result, args.out_file, sensitivity, zero_sensitivity_layers):
        print(f"Wrote {path}")
    _update_sweep_summary(args.out_file.parent, args, result, zero_sensitivity_layers)


if __name__ == "__main__":
    main()
