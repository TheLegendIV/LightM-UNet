"""Assemble a whole-network folding from the ANALYTICAL per-block models, written in the MILP flow's own formats.

    python3 MILP/analytical/net_fold.py --config config_12_dense_relu_nearest_upsample_256 --bits 6 --fps 250 --clock-mhz 100 \\
        --tag U4_int6_fps250 --out-dir MILP/artifacts/U4_256_int6_fps250_analytical_v1

Pipeline (the MILP's own plumbing is reused, nothing is re-derived):
  1. finn_milp.build_model_and_graph() traces the network (LayerQuantEnetFINN has the same layer / site names as ENet) -> LayerGeometry per conv / pool,
     ExtraNode per join / dup / threshold, and the dataflow graph.
  2. Blocks are visited in dataflow order. Each one runs its analytical model (model_int_bottleneck / model_dn_bottleneck / model_bottleneck /
     model_up_bottleneck / model_fnl_block) at the block's frame-cycle budget, which gives PE / SIMD of every MVAU, the PE of the thresholds, the
     joins (Add, residual / out_act merged, skip), the pad-MVAU of the downsampling skip, and the PE rule of the nearest-upsampler join chain.
  3. Those folds are written under the MILP layer / extra-node names with the MILP cost functions (layer_cost_pe_simd, extra_node_options), so
     per_layer / extra_nodes / _diagnostics / dataflow_graph have exactly the schema of finn_milp.py's output.
     expand_layer_bits.py then expands the bits to the per-quantizer-site file (layer_bits_SITES_<tag>.json).
  4. Blocks are realised and their INTERNAL FIFOs sized to minimum memory (each block's verify_with_sim; stored as top-level `intra_block_fifos`, treated as a
     black box from here on). Then the INTER-block FIFOs (top-level `inter_block_fifos`, one per dataflow edge that leaves a block, each block owns the FIFO at its
     output) are all fixed at depth 2 by default (--inter-fifo-depth; FINN's RemoveShallowFIFOs deletes such FIFOs); --whole-net-check optionally simulates the chain to confirm the target.
     --inter-fifo sim sizes them instead with net_fifo.py: block i + FIFO(D) + block i+1 simulated together with a saturated source, smallest D whose
     last-frame period meets --fifo-target-fps (default --fps), then the whole-net check. --inter-fifo standard: one BRAM18 per edge (6 bit x 2048), no simulation.
  5. enet_dataflow_<tag>.onnx: every hardware node, DWC and FIFO of the sized design as one ONNX graph (net_onnx.py) for a visual check in Netron.

Rate policy:
  * every node's cycles per frame <= F = clock / fps (250 fps at 100 MHz = 400,000 cycles); the analytical search takes the largest cycle count that
    fits, so each stage is matched to the target;
  * optional latency cap --max-latency-ms (latency = cycles to the FIRST OUTPUT PIXEL, summed over the blocks' first-in -> first-out latencies / clock): the per-node budget is bisected down until it holds;
  * "downstream faster than upstream": block k gets budget min(F, max(floor * F, R * slowest_{k-1})), R = 1 + --ratchet-pct/100 (the MILP's --ratchet-pct, the same rule edge by edge on the dataflow graph; the element-rate DSR is a different, opt-in constraint; the MILP's own DSR allowance, default
    1.04), floor = --ratchet-floor. A node that cannot be slowed (the lattice of legal (PE, SIMD) pairs) keeps the chain from relaxing past the floor,
    and the achieved MILP DSR ratio (cycles per output element, downstream vs. upstream) is reported in _diagnostics.chain_rate_imbalance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "enet", REPO / "MILP", REPO / "MILP" / "utils", REPO / "MILP" / "analytical"):
    sys.path.insert(0, str(p))

import finn_milp  # noqa: E402
import finn_cost_model as fcm  # noqa: E402
from finn_cost_model import calibrated_bram18k, calibrated_lut, layer_cost_pe_simd_auto_ram  # noqa: E402
from finn_milp import (  # noqa: E402
    EXTRA_OP_LABEL, RAM_STYLE_AUTO, STREAM_NODE_KINDS, THRESHOLD_KINDS, VARIANT_RTL_DSP_NOACT1, XCZU7EV, _calibration_force_dsp, _variant_cost_kwargs,
    build_model_and_graph, extra_node_options,
)
from milp_outputs import compute_branch_imbalance_report, compute_chain_rate_imbalance_report  # noqa: E402

from node_names import block_output_name, milp_role  # noqa: E402
from bottleneck import _retarget_threshold, model_bottleneck  # noqa: E402
from bottleneck import to_folding_config as _fold_reg, verify_with_sim as _verify_reg  # noqa: E402
from dn_bottleneck import model_dn_bottleneck  # noqa: E402
from dn_bottleneck import to_folding_config as _fold_dn, verify_with_sim as _verify_dn  # noqa: E402
from fnl_block import model_fnl_block  # noqa: E402
from fnl_block import to_folding_config as _fold_final, verify_with_sim as _verify_final  # noqa: E402
from int_bottleneck import model_int_bottleneck  # noqa: E402
from int_bottleneck import to_folding_config as _fold_init, verify_with_sim as _verify_init  # noqa: E402
from up_bottleneck import model_up_bottleneck  # noqa: E402
from up_bottleneck import to_folding_config as _fold_up, verify_with_sim as _verify_up  # noqa: E402

# per-block-kind (verify_with_sim, to_folding_config) pair -- same schema from every module (see each
# module's own to_folding_config docstring: "same schema as bottleneck.to_folding_config").
_VERIFY_FOLD = {
    "reg": (_verify_reg, _fold_reg), "dn": (_verify_dn, _fold_dn), "up": (_verify_up, _fold_up),
    "init": (_verify_init, _fold_init), "final": (_verify_final, _fold_final),
}

_XOPT_CACHE: dict = {}


def xopts(node):
    """finn_milp.extra_node_options(node, force_dsp=True), cached per node (it is called for every node on every budget trial)."""
    key = (node.geom.name, finn_milp.CANDIDATE_BITS)
    if key not in _XOPT_CACHE:
        _XOPT_CACHE[key] = extra_node_options(node, True)
    return _XOPT_CACHE[key]


THR_RAM_STYLE = "block"              # the analytical models price every standalone threshold with ram_style="block" (depth_trigger_bram 1024 in the probes)
_BRAM18_ASPECTS = ((1, 16384), (2, 8192), (4, 4096), (9, 2048), (18, 1024), (36, 512))   # UG573 table 1-10 (width, depth)
FOLDABLE = ("MVAU", "Thr", "Add", "Dup", "Label")


# ---------------------------------------------------------------------------------------------- block bookkeeping
def block_order(geoms) -> list[str]:
    seen: list[str] = []
    for g in geoms:
        if g.stage not in seen:
            seen.append(g.stage)
    return seen


def block_kind(stage: str) -> str:
    if stage == "initial":
        return "init"
    if stage == "final":
        return "final"
    if stage.startswith("down"):
        return "dn"
    if stage.startswith("up"):
        return "up"
    return "reg"


def run_block(stage: str, geom: dict, bits: int, F_k: int, compute_fifos: bool = False, prev_output: str | None = None):
    """Analytical model of one block at budget F_k cycles per frame -> (BlockResult, layer folds, extra folds,
    slowest, intra-block FIFOs, FIFO-verification warning). compute_fifos runs verify_with_sim + to_folding_config
    on top of the same chosen BottleneckResult (same PE/SIMD/bits as lf/xf) to also get this block's OWN FIFO depths
    (skip FIFO, FMPad prefetch, ...) -- off by default since it is only needed once per final config, not on every
    bisection trial in main()'s latency-cap search."""
    kind = block_kind(stage)
    lf: dict[str, tuple] = {}     # MILP layer name -> (pe, simd, thr_pe or None, analytical frame cycles of the MVAU)
    xf: dict[str, tuple] = {}     # MILP extra node name -> (pe, simd, ram_style or None)

    def put(layer, mv, thr=None):
        lf[layer] = (mv.pe, mv.simd, thr.pe if thr is not None else None, mv.frame_cycles)

    if kind == "reg":
        r0 = geom[f"{stage}.reduce.0"]
        cin, cmid = r0.cin, r0.cout
        d = geom[f"{stage}.conv"].dh
        px = r0.hin * r0.win
        r = model_bottleneck(cin, cin // cmid, cin // cmid, max(1, F_k // px), bits, r0.hin, r0.win, k=3, dilation=d)
        n = {x.name: x for x in r.nodes}
        put(f"{stage}.reduce.0", n["MVAU_r"], n["Thr_r"]); put(f"{stage}.conv", n["MVAU_m"], n["Thr_m"]); put(f"{stage}.expand.0", n["MVAU_e"], n["Thr_e"])
        xf[f"{stage}.add"] = (n["Add"].pe, 1, None)
        xf[f"{stage}.skip_quant"] = (n["Thr_s"].pe, 1, THR_RAM_STYLE)
        xf[f"{stage}.residual_add"] = (n["Thr_out"].pe, 1, THR_RAM_STYLE)
        xf[f"{stage}.out_act"] = (n["Thr_out"].pe, 1, THR_RAM_STYLE)
    elif kind == "dn":
        pool = geom[f"{stage}.pool"]
        cin, cout = pool.cin, geom[f"{stage}.expand.0"].cout
        cmid = geom[f"{stage}.reduce.0"].cout
        r = model_dn_bottleneck(cin, cout, cout // cmid, bits, pool.hin, pool.win, T_out=F_k / (pool.hout * pool.wout), skip_order="pad_thr", skip_pad="mvau")
        n = {x.name: x for x in r.nodes}
        put(f"{stage}.reduce.0", n["MVAU_r"], n["Thr_r"]); put(f"{stage}.conv.0", n["MVAU_m"], n["Thr_m"]); put(f"{stage}.expand.0", n["MVAU_e"], n["Thr_e"])
        xf[f"{stage}.skip_pad"] = (n["MVAU_s"].pe, n["MVAU_s"].simd, "auto")
        xf[f"{stage}.skip_quant"] = (n["Thr_s"].pe, 1, THR_RAM_STYLE)
        xf[f"{stage}.add"] = (n["Add"].pe, 1, None)
        xf[f"{stage}.residual_add"] = (n["Thr_out"].pe, 1, THR_RAM_STYLE)
        xf[f"{stage}.out_act"] = (n["Thr_out"].pe, 1, THR_RAM_STYLE)
    elif kind == "up":
        mp = geom[f"{stage}.main_proj.0"]
        cin, cout, cmid = mp.cin, mp.cout, geom[f"{stage}.reduce.0"].cout
        r = model_up_bottleneck(cin, cout, cin // cmid, bits, mp.hin, mp.win, T_out=F_k / (4 * mp.hin * mp.win), skip_conv=False)
        n = {x.name: x for x in r.nodes}
        put(f"{stage}.main_proj.0", n["MVAU_p"], n["Thr_p"]); put(f"{stage}.reduce.0", n["MVAU_r"], n["Thr_r"])
        put(f"{stage}.up.0", n["MVAU_u"], n["Thr_u"]); put(f"{stage}.expand.0", n["MVAU_e"], n["Thr_e"])
        xf[f"{stage}.skip_quant"] = (n["Thr_s"].pe, 1, THR_RAM_STYLE)
        xf[f"{stage}.add"] = (n["Add"].pe, 1, None)
        xf[f"{stage}.residual_add"] = (n["Thr_out"].pe, 1, THR_RAM_STYLE)
        xf[f"{stage}.out_act"] = (n["Thr_out"].pe, 1, THR_RAM_STYLE)
    elif kind == "init":
        c = geom["initial.conv"]
        r = model_int_bottleneck(c.cin, c.cin + c.cout, bits, c.hin, c.win, F=F_k)
        n = {x.name: x for x in r.nodes}
        put("initial.conv", n["MVAU_c"], n["Thr_c"])
        xf["initial.input_quant"] = (n["Thr_in"].pe, 1, THR_RAM_STYLE)
        xf["initial.act"] = (n["Thr_act"].pe, 1, THR_RAM_STYLE)
        xf["initial.pool_quant"] = (n["Thr_m"].pe, 1, THR_RAM_STYLE)       # the branch-quant threshold UPSTREAM of the maxpool (MILP kind pool_quant)
    else:   # final: the real build (partition 7) has the bias ChannelwiseOp (INT16 -> INT17, PE 1) and a LabelSelect argmax behind the MVAU; neither the bias
        # (no MILP node yet) nor the intra-block DWC / FIFO nodes are MILP entries, only final.argmax is (kind argmax)
        f = geom["final"]
        r = model_fnl_block(f.cin, f.cout, bits, f.hin, f.win, F=F_k, bias=True, argmax=True)
        n = {x.name: x for x in r.nodes}
        lf["final"] = (n["MVAU_f"].pe, n["MVAU_f"].simd, None, n["MVAU_f"].frame_cycles)
        xf["final.argmax"] = (n["LabelSelect"].pe, 1, None)
    slowest = max(x.frame_cycles for x in r.nodes if x.name.startswith(FOLDABLE))
    fifos, fifo_warning = (_intra_block_fifos(kind, r, stage, prev_output) if compute_fifos else ([], None))
    return r, lf, xf, slowest, fifos, fifo_warning


def _intra_block_fifos(kind: str, r, stage: str, prev_output: str | None = None) -> tuple[list[dict], str | None]:
    """verify_with_sim(r) + to_folding_config(r)['fifos'], stage-qualified (role names -> '<stage>.<role>') so every
    block's FIFOs land in one flat, uniquely-named list. Soft-fails (returns [] + a warning string) if the simulation
    does not converge -- same pattern as the mvau_cycle_mismatches diagnostic, surfaced not silently dropped."""
    verify_fn, _ = _VERIFY_FOLD[kind]
    try:
        verify_fn(r)
    except RuntimeError as e:
        return [], f"{stage}: verify_with_sim failed -- {e}"
    return _fifo_entries(kind, r, stage, prev_output), None


def _fifo_entries(kind: str, r, stage: str, prev_output: str | None = None) -> list[dict]:
    """Flat intra-block FIFO list of an ALREADY VERIFIED block result (verify_with_sim ran): the schema the FINN bridge reads, plus the MILP names."""
    cfg = _VERIFY_FOLD[kind][1](r)
    fifos = [
        {
            "stage": stage, "name": f"{stage}.{f['name'].replace(' ', '_')}",
            "producer": f"{stage}.{f['producer']}", "consumer": f"{stage}.{f['consumer']}",
            "producer_node": f["producer_node"], "consumer_node": f["consumer_node"],
            "producer_milp": milp_role(stage, kind, f["producer_node"], prev_output), "consumer_milp": milp_role(stage, kind, f["consumer_node"], prev_output),
            "depth": f["depth"], "width_bits": f["width_bits"], "max_occupancy": f["max_occupancy"],
            "is_skip": f["is_skip"], "mem": f.get("mem"), "finn_impl": f.get("finn_impl"),
            "mem_bram18": f.get("mem_bram18", 0), "mem_lut": f.get("mem_lut", 0),
        }
        for f in cfg["fifos"]
    ]
    return fifos


# ---------------------------------------------------------------------------------------------- MILP-format entries
def layer_entry(geom, bits, pe, simd, thr_pe, stage, wbits=None, no_thr=False):
    """per_layer entry priced EXACTLY like the analytical models price a layer: layer_cost_pe_simd_auto_ram(force_dsp=True), then the standalone threshold
    re-targeted to the analytical PE_t (bottleneck._retarget_threshold, ram_style block). no_thr: the layer has no threshold (the final logits layer)."""
    wb = wbits or bits
    if geom.op_type == "MaxPool2d":
        pe, simd = 1, 1
        cost = layer_cost_pe_simd_auto_ram(geom, wb, bits, 1, 1, force_dsp=True)
    else:
        cost = layer_cost_pe_simd_auto_ram(geom, wb, bits, pe, simd, force_dsp=True)
        if thr_pe is not None:
            cost = _retarget_threshold(cost, geom, bits, thr_pe)
    cost = {k: v for k, v in cost.items() if k != "ram_style_chosen"}
    if no_thr:
        cost = {**cost, "total_lut": cost["total_lut"] - cost["thr_lut"], "thr_lut": 0.0, "thr_bram18": 0.0, "thr_uram18": 0.0}
    vk = _variant_cost_kwargs(VARIANT_RTL_DSP_NOACT1, True)
    e = {"stage": stage, "pe": pe, "simd": simd, "thr_ram_style": THR_RAM_STYLE, "variant": VARIANT_RTL_DSP_NOACT1, "force_dsp": vk["force_dsp"],
         "mvau_noAct": vk["no_activation"], "weight_bits": wb, "act_bits": bits, **cost}
    e["lut_calibrated"] = calibrated_lut(e["total_lut"], wb, bits, force_dsp=_calibration_force_dsp(vk))
    e["bram18k_calibrated"] = calibrated_bram18k(e["swu_bram18"] + e["wm_bram18"] + e.get("thr_bram18", 0), wb, bits, force_dsp=True)
    return e


def _x_entry(node, pe, simd, ram_style, bits, cycles, lut, bram, dsp=0, uram=0.0, w=None, a=None, **extra):
    return {
        "kind": node.kind, "stage": node.geom.stage, "pe": pe, "simd": simd, "ram_style": ram_style, "weight_bits": w, "act_bits": a,
        "bit_sources": list(node.bit_sources), "channels": node.geom.cout, "cycles": cycles, "lut_calibrated": lut, "bram18k_calibrated": bram,
        "dsp": dsp, "uram18": uram,
        # dup's own MILP name is producer-qualified ('<producer>.dup', see layer_topology.DUP_SUFFIX) while its
        # stage is the CONSUMING block -- every other extra node's name already equals "<stage>.<kind>", so this
        # is a no-op alias for them and the real fix only for dup. Consumers (e.g. the FINN FIFO bridge) should
        # match role names through this field instead of each inventing their own name-mangling convention.
        "canonical_role": f"{node.geom.stage}.{node.kind}", **extra,
    }


def pad_mvau_cost(node, bits, pe, simd):
    """The downsampling skip's 1x1 pad-MVAU as dn_bottleneck prices it: INT8 weights, network activation bits, its threshold (PE_t = Thr_s PE) belongs to the
    skip threshold node, not to the MVAU."""
    return layer_cost_pe_simd_auto_ram(node.geom, 8, bits, pe, simd, force_dsp=True)


def extra_entry(node, bits, pe, simd, ram_style, pad_cost=None):
    """Extra-node entry priced like the analytical models: threshold_node_cost (block RAM, default input width) for every standalone threshold, stream_node_cost
    for dup / add / concat / upsample. residual_add is MERGED into out_act (the compose pass; analytical Thr_out): kept in the graph at zero cost."""
    kind, g = node.kind, node.geom
    rk = _variant_cost_kwargs(VARIANT_RTL_DSP_NOACT1, True)
    if kind == "residual_add":
        return _x_entry(node, pe, simd, ram_style, bits, 0, 0.0, 0.0, a=bits, merged_into=g.name[: -len("residual_add")] + "out_act")
    if kind == "pad_mvau":
        c = pad_cost
        return _x_entry(node, pe, simd, "auto", bits, c["mvu_cycles"], calibrated_lut(c["total_lut"] - c["thr_lut"], 8, bits, force_dsp=_calibration_force_dsp(rk)),
                        calibrated_bram18k(c["swu_bram18"] + c["wm_bram18"], 8, bits, force_dsp=rk["force_dsp"]), dsp=c["total_dsp"],
                        uram=c["wm_uram18"] + c.get("swu_uram18", 0), w=8, a=bits, threshold_in="skip_quant")
    if kind == "skip_quant" and pad_cost is not None:         # downsampling skip: the threshold retargeted in the pad-MVAU's own cost (in_bits = its accumulator)
        lut, bram, uram = fcm._thresholding_rtl_cost(pe, bits, g.cout, ram_style="block", in_bits=pad_cost["acc_bits"])
        return _x_entry(node, pe, 1, "block", bits, g.hout * g.wout * math.ceil(g.cout / pe), calibrated_lut(lut, bits, bits, force_dsp=_calibration_force_dsp(rk)),
                        calibrated_bram18k(bram, bits, bits, force_dsp=rk["force_dsp"]), uram=uram, a=bits)
    if kind in THRESHOLD_KINDS:
        c = fcm.threshold_node_cost(g, bits, pe)
        return _x_entry(node, pe, 1, "block", bits, c["cycles"], calibrated_lut(c["total_lut"], bits, bits, force_dsp=_calibration_force_dsp(rk)),
                        calibrated_bram18k(c["thr_bram18"], bits, bits, force_dsp=rk["force_dsp"]), uram=c["thr_uram18"], a=bits)
    c = fcm.stream_node_cost(kind, g, pe) if kind in fcm.FOLDABLE_STREAM_KINDS else fcm.stream_node_cost(kind, g)
    return _x_entry(node, pe if kind in fcm.FOLDABLE_STREAM_KINDS else 1, 1, "none", bits, c["cycles"], c["total_lut"], c["swu_bram18"] + c["wm_bram18"] + c["thr_bram18"])


def best_extra_fold(node, bits, budget):
    """Fallback (dup, concat, upsample, anything not mapped from an analytical node): the legal option with the largest cycle count <= budget, then cheapest."""
    opts = [(k, c) for k, c in xopts(node) if node.kind not in THRESHOLD_KINDS or k[6] == bits]
    fit = [(k, c) for k, c in opts if c["cycles"] <= budget] or [min(opts, key=lambda kc: kc[1]["cycles"])]
    k, c = max(fit, key=lambda kc: (kc[1]["cycles"], -kc[1]["total_lut"]))
    return k[1], k[2], k[3]


def bram_fifo(width_bits: int) -> dict:
    """One-BRAM18 standard FIFO: the narrowest SDP aspect that holds the stream width gives the depth; wider streams span several BRAM18 at depth 512."""
    for w, d in _BRAM18_ASPECTS:
        if w >= width_bits:
            return {"depth": d, "bram18": 1, "aspect": [w, d]}
    w, d = _BRAM18_ASPECTS[-1]
    return {"depth": d, "bram18": math.ceil(width_bits / w), "aspect": [w, d]}


def write_sites(config: str, folding_json: Path, sites_json: Path) -> None:
    """layer_bits_SITES_<tag>.json = MILP/expand_layer_bits.py run in-process. The working-tree LayerQuantENet.py lacks its `from QuantENet import (...)`
    block, VALID_CONTEXT_PATTERNS and ACT_SITE_TYPES (dropped by the 'Sync' commit 66e0590e8b, present at b16b35468a; the file is not touched here), so what it needs from QuantENet (helpers, VALID_CONTEXT_PATTERNS, the dilation patterns) is injected first, as the probe export scripts do."""
    import nnunetv2.nets.LayerQuantENet as lq
    import nnunetv2.nets.CombinedQuantENet as cq
    import nnunetv2.nets.ENet as en
    import nnunetv2.nets.QuantENet as qe
    import brevitas.nn as qnn
    for mod in (qe, cq, en):      # QuantENet helpers, CombinedQuantENet.VALID_CONTEXT_PATTERNS, ENet's dilation patterns
        for name in dir(mod):
            if not name.startswith("__") and not hasattr(lq, name):
                setattr(lq, name, getattr(mod, name))
    if not hasattr(lq, "ACT_SITE_TYPES"):      # as defined in LayerQuantENet.py at b16b35468a (the later "Sync" commit dropped it)
        lq.ACT_SITE_TYPES = (qnn.QuantReLU, qnn.QuantIdentity, qe.QuantDecomposedLeakyAct, qe.QuantFusedLeakyAct, qnn.QuantEltwiseAdd)
    import expand_layer_bits
    old_argv = sys.argv
    sys.argv = ["expand_layer_bits.py", "--config", config, "--ilp-result", str(folding_json), "--out-file", str(sites_json)]
    try:
        expand_layer_bits.main()
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------------------------- assembly
def assemble(a, ctx, F_top: int, compute_intra_fifos: bool = False, keep_blocks: bool = False):
    """One whole-network assembly with per-node budget F_top (cycles per frame): the analytical models per block (optionally ratcheted so that every
    block is no slower than the one before it), costed with the MILP functions. Returns the MILP-layout result dict.
    compute_intra_fifos additionally verifies+sizes every block's OWN (skip / prefetch / ...) FIFOs via each block module's
    verify_with_sim + to_folding_config, flattened into result["intra_block_fifos"] -- only pass True on the final chosen
    F_top (expensive per block, and pointless to repeat on every latency-cap bisection trial in main()). keep_blocks
    stashes result["_blocks"] = [(stage, kind, analytical result)] for the richer post-assembly pipeline (verify_blocks /
    intra_block_report / size_inter_block_fifos), which supersedes the compute_intra_fifos output when both run."""
    geoms, extras, geom, xnode, dmap, kinds = ctx
    per_layer: dict[str, dict] = {}
    extra_out: dict[str, dict] = {}
    profile = []
    slowest_prev = None
    mismatches = []
    rough_latency = 0
    intra_fifos: list[dict] = []
    intra_fifo_warnings: list[str] = []
    blocks = []
    prev_output = None
    for stage in block_order(geoms):
        kind = block_kind(stage)
        F_k = F_top if (a.no_ratchet or slowest_prev is None) else int(min(F_top, max(a.ratchet_floor * F_top, (1 + a.ratchet_pct / 100) * slowest_prev)))
        r, lf, xf, slowest, fifos, fifo_warning = run_block(stage, geom, a.bits, F_k, compute_fifos=compute_intra_fifos, prev_output=prev_output)
        intra_fifos.extend(fifos)
        if fifo_warning:
            intra_fifo_warnings.append(fifo_warning)
        rough_latency += r.latency_first_out_cycles
        blocks.append((stage, kind, r))
        prev_output = block_output_name(stage, kind)
        for lname, (pe, simd, thr_pe, an_cycles) in lf.items():
            e = layer_entry(geom[lname], a.bits, pe, simd, thr_pe, stage, no_thr=(lname == "final"))
            if e["mvu_cycles"] != an_cycles:
                mismatches.append((lname, an_cycles, e["mvu_cycles"]))
            per_layer[lname] = e
        for lg in (g for g in geoms if g.stage == stage and g.name not in per_layer):      # layers the analytical block has no node for (maxpool)
            per_layer[lg.name] = layer_entry(lg, a.bits, 1, 1, None, stage)
        pad_cost = None
        if f"{stage}.skip_pad" in xf:                    # downsampling skip: the pad-MVAU's cost also carries the skip threshold's accumulator width
            ppe, psimd, _ = xf[f"{stage}.skip_pad"]
            pad_cost = pad_mvau_cost(xnode[f"{stage}.skip_pad"], a.bits, ppe, psimd)
        for xn in (n for n in extras if n.geom.stage == stage):
            pe, simd, rs = xf.get(xn.geom.name) or best_extra_fold(xn, a.bits, F_k)
            extra_out[xn.geom.name] = extra_entry(xn, a.bits, pe, simd, rs, pad_cost=pad_cost if xn.kind in ("pad_mvau", "skip_quant") else None)
            extra_out[xn.geom.name]["bits_rule"] = "fixed" if xn.fixed_bits is not None else ("max_of_sources" if xn.bit_sources else "n/a")
        profile.append(dict(block=stage, kind=kind, budget=F_k, slowest_foldable=slowest, ratio_to_prev=(round(slowest / slowest_prev, 3) if slowest_prev else None)))
        slowest_prev = slowest
    for xn in extras:       # fork-source dups named after a previous block's out_act can belong to no block above
        if xn.geom.name not in extra_out:
            pe, simd, rs = best_extra_fold(xn, a.bits, F_top)
            extra_out[xn.geom.name] = extra_entry(xn, a.bits, pe, simd, rs)
            extra_out[xn.geom.name]["bits_rule"] = "fixed" if xn.fixed_bits is not None else ("max_of_sources" if xn.bit_sources else "n/a")

    # ---- inter-block FIFOs: one per dataflow edge that leaves a block
    def stage_of(n):
        return per_layer[n]["stage"] if n in per_layer else extra_out[n]["stage"]

    def width_of(n):
        if n in extra_out:
            return extra_out[n]["pe"] * a.bits
        return (geom[n].cout if geom[n].op_type == "MaxPool2d" else per_layer[n].get("thr_pe", 1)) * a.bits
    known = per_layer.keys() | extra_out.keys()
    fifos = []
    for name, preds in dmap.items():
        for pr in preds:
            if pr in known and name in known and stage_of(pr) != stage_of(name):
                w = width_of(pr)
                bf = bram_fifo(w)
                fifos.append({"name": f"{pr}->{name}", "producer": pr, "consumer": name, "src_block": stage_of(pr), "dst_block": stage_of(name),
                              "width_bits": w, "mem": "bram", **bf,
                              "efficiency": round(w * bf["depth"] / (bf["bram18"] * 18432), 3)})

    # ---- totals / diagnostics in the MILP layout
    L, X = list(per_layer.values()), list(extra_out.values())
    total_lut = sum(v["lut_calibrated"] for v in L + X)
    total_bram = sum(v["bram18k_calibrated"] for v in L + X)
    total_uram = sum(v.get("wm_uram18", 0) + v.get("swu_uram18", 0) + v.get("thr_uram18", 0) for v in L) + sum(v["uram18"] for v in X)
    total_dsp = sum(v["total_dsp"] for v in L) + sum(v["dsp"] for v in X)
    total_cycles = sum(v["cycles"] for v in L + X)
    fifo_bram = sum(f["bram18"] for f in fifos)
    intra_fifo_bram = sum(f["mem_bram18"] for f in intra_fifos)
    node_cycles = {**{n: v["cycles"] for n, v in per_layer.items()}, **{n: v["cycles"] for n, v in extra_out.items()}}
    bott = max(node_cycles, key=node_cycles.get)
    hardware_nodes = [*geoms, *(n.geom for n in extras)]
    result = {
        "status": "Optimal",
        "layer_weight_bits": {n: a.bits for n in per_layer},
        "layer_act_bits": {n: a.bits for n in per_layer},
        "per_layer": per_layer,
        "extra_nodes": extra_out,
        "inter_block_fifos": fifos,
        "intra_block_fifos": intra_fifos,
        "_diagnostics": {
            "n_layers": len(per_layer), "n_extra_nodes": len(extra_out), "force_dsp": True, "min_resources": True,
            "total_lut_calibrated": total_lut, "xczu7ev_lut_budget": XCZU7EV["LUT"], "lut_pct_of_budget": 100 * total_lut / XCZU7EV["LUT"],
            "total_bram18k_calibrated": total_bram, "xczu7ev_bram18k_budget": XCZU7EV["BRAM_18K"], "bram_pct_of_budget": 100 * total_bram / XCZU7EV["BRAM_18K"],
            "inter_block_fifo_bram18": fifo_bram, "intra_block_fifo_bram18": intra_fifo_bram,
            "total_bram18k_with_inter_block_fifos": total_bram + fifo_bram,
            "total_bram18k_with_all_fifos": total_bram + fifo_bram + intra_fifo_bram,
            "intra_block_fifo_warnings": intra_fifo_warnings,
            "total_uram18": total_uram, "xczu7ev_uram_budget": XCZU7EV["URAM"], "uram_pct_of_budget": 100 * total_uram / XCZU7EV["URAM"],
            "total_cycles": total_cycles, "sum_of_node_cycles_ms": total_cycles / (a.clock_mhz * 1e3),
            "latency_first_out_cycles": rough_latency, "latency_ms": rough_latency / (a.clock_mhz * 1e3),
            "total_dsp": total_dsp, "xczu7ev_dsp_budget": XCZU7EV["DSP"], "dsp_pct_of_budget": 100 * total_dsp / XCZU7EV["DSP"],
            "bottleneck_node": bott, "bottleneck_cycles": node_cycles[bott],
            "max_node_cycles": int(a.clock_mhz * 1e6 / a.fps), "per_node_budget_used": F_top, "target_fps": a.fps, "clock_mhz": a.clock_mhz,
            "max_latency_ms": a.max_latency_ms, "fps": a.clock_mhz * 1e6 / node_cycles[bott],
            "analytical": {"source": "MILP/analytical/net_fold.py", "ratchet_pct": a.ratchet_pct, "ratchet_floor": a.ratchet_floor, "mvau_wwidth_max": a.mvau_wwidth_max, "ratchet": not a.no_ratchet,
                           "sum_of_block_first_out_latencies_cycles": rough_latency, "block_profile": profile,
                           "mvau_cycle_mismatches_vs_milp_cost_model": mismatches},
            "note": "Folding assembled from the analytical per-block models (U4 widths), costed with the MILP's own layer_cost_pe_simd / extra_node_options. Not an ILP solve. "
                    "latency_ms = cycles from the first input pixel to the first output pixel (sum over blocks of each block model's first-in -> first-out latency / clock); "
                    "total_cycles / sum_of_node_cycles_ms is the MILP's work-per-frame measure (every node's cycles summed), kept for the MILP schema, NOT a latency.",
        },
        "dataflow_graph": {
            "input": [finn_milp.IN_CHANNELS, *finn_milp.INPUT_HW], "edges": dmap,
            "shapes": {g.name: [g.cout, g.hout, g.wout] for g in hardware_nodes},
            "op_types": {**{g.name: g.op_type for g in geoms}, **{n.geom.name: EXTRA_OP_LABEL[n.kind] for n in extras}},
            "kinds": kinds,
        },
    }
    if keep_blocks:
        result["_blocks"] = blocks
    fixed = sorted([n for n, g in geom.items() if g.op_type == "MaxPool2d"] + [n for n, x in xnode.items() if x.kind in ("concat", "upsample")])
    result["_diagnostics"]["fixed_cycle_nodes"] = fixed
    return result


# ---------------------------------------------------------------------------------------------- FIFOs: realise blocks, walk the net
def _verify_task(args):
    kind, r, stage = args
    import bottleneck, dn_bottleneck, fnl_block, int_bottleneck, up_bottleneck
    ver = {"reg": bottleneck, "dn": dn_bottleneck, "up": up_bottleneck, "init": int_bottleneck, "final": fnl_block}[kind].verify_with_sim
    try:
        ver(r)
    except RuntimeError as e:
        raise RuntimeError(f"block {stage}: {e}") from None
    return r


def verify_blocks(blocks: list, workers: int) -> list:
    """Step 2 of the flow: realise every block (its folding is already fixed) and size its INTERNAL FIFOs to minimum memory (verify_with_sim). Blocks with the
    same shape and folding are verified once."""
    key = lambda kind, r: repr((kind, sorted(r.params.items()), [(n.name, n.pe, n.simd) for n in r.nodes]))
    todo: dict = {}
    for stage, kind, r in blocks:
        todo.setdefault(key(kind, r), (kind, r, stage))
    print(f"realising {len(blocks)} blocks ({len(todo)} distinct shapes) and sizing their internal FIFOs, {workers} workers", flush=True)
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            done = pool.map(_verify_task, list(todo.values()))
    else:
        done = [_verify_task(t) for t in todo.values()]
    ver = dict(zip(todo, done))
    return [(stage, kind, ver[key(kind, r)]) for stage, kind, r in blocks]


def size_inter_block_fifos(result: dict, blocks: list, a, F_target: int) -> list:
    """Step 3: walk the net pair by pair (net_fifo.size_interfaces), then one whole-net run as a closing check (all interface depths doubled until it holds).
    Rewrites result["inter_block_fifos"] / diagnostics; returns [dict(depth, width_bits, max_occ, period)] per interface for the ONNX export."""
    import net_fifo
    from bottleneck import fifo_memory
    fixed = a.inter_fifo == "fixed"
    if fixed:
        print(f"{len(blocks) - 1} inter-block FIFOs fixed at depth {a.inter_fifo_depth}", flush=True)
        sized = [dict(depth=a.inter_fifo_depth, status="fixed", period=None, max_occ=0, sims=0, tested={}) for _ in range(len(blocks) - 1)]
    else:
        print(f"sizing {len(blocks) - 1} inter-block FIFOs by pair simulation: smallest depth with last-frame period <= {F_target * (1 + a.fifo_slack):.0f} cycles/frame", flush=True)
        sized = net_fifo.size_interfaces(blocks, F_target, a.fifo_slack, workers=a.workers)
    scale, whole = 1, None
    if a.whole_net_check:
        while True:
            depths = [max(net_fifo.MIN_DEPTH, s["depth"] * scale) for s in sized]
            print(f"whole-net check with interface depths x{scale} ...", flush=True)
            whole = net_fifo.run_whole(blocks, depths, F_target)
            print(f"  last-frame period {whole['period']} (target {F_target}), deadlock {whole['deadlock']}", flush=True)
            if whole["ok"] or scale >= 8 or fixed:
                break
            scale *= 2
        if fixed:
            for s_, occ in zip(sized, whole["iface_max"]):
                s_["max_occ"] = occ
            if not whole["ok"]:
                print(f"WARNING: whole-net run misses the target with every inter-block FIFO fixed at depth {a.inter_fifo_depth}", flush=True)
        whole = {k: whole[k] for k in ("ok", "deadlock", "period", "periods", "first_out", "target")} | {"depth_scale": scale}
    else:
        depths = [s["depth"] for s in sized]
    by_src = {f["src_block"]: f for f in result["inter_block_fifos"]}
    out, unmatched = [], 0
    for i, s in enumerate(sized):
        e = by_src.get(blocks[i][0])
        if e is None or e["dst_block"] != blocks[i + 1][0]:
            unmatched += 1
            continue
        d = depths[i]
        m = fifo_memory(e["width_bits"], d)
        e.pop("aspect", None)
        e.update(depth=d, mem=m["mem"], depth_alloc=int(m["depth_alloc"]), bram18=int(m["bram18"]), lut=int(m["lut"]), uram18=int(m["uram"]),
                 efficiency=round(m["efficiency"], 3), max_occ=s["max_occ"], sizing=("fixed" if fixed else "pair_simulation"), sizing_status=s["status"], pair_period_cycles=s["period"])
        out.append(dict(depth=d, width_bits=e["width_bits"], max_occ=s["max_occ"], period=s["period"]))
    fl = result["inter_block_fifos"]
    dg = result["_diagnostics"]
    dg["inter_block_fifo_bram18"] = sum(f["bram18"] for f in fl)
    dg["inter_block_fifo_lut"] = sum(f.get("lut", 0) for f in fl)
    dg["inter_block_fifo_uram18"] = sum(f.get("uram18", 0) for f in fl)
    dg["total_bram18k_with_inter_block_fifos"] = dg["total_bram18k_calibrated"] + dg["inter_block_fifo_bram18"]
    dg["total_bram18k_with_all_fifos"] = dg["total_bram18k_with_inter_block_fifos"] + dg.get("intra_block_fifo_bram18", 0)
    dg["fifo_sizing"] = dict(
        method=(f"every inter-block FIFO fixed at depth {a.inter_fifo_depth}" if fixed else
                "pair simulation: block i (saturated input, verified intra-block FIFOs replayed unchanged) -> FIFO(D) -> block i+1; smallest D with last-frame "
                "period <= target * (1 + slack); then a whole-net run"), target_cycles_per_frame=F_target, slack=a.fifo_slack, min_depth=net_fifo.MIN_DEPTH,
        pairs=[dict(src=blocks[i][0], dst=blocks[i + 1][0], **s) for i, s in enumerate(sized)], whole_net=whole, unmatched_interfaces=unmatched)
    return out


def intra_block_report(blocks: list, result: dict) -> None:
    """Store the per-block verification summary (verified depths of the internal FIFOs above 2, DWCs, warnings) under `block_verification`."""
    rep, tot = {}, dict(fifo_lut=0.0, fifo_bram18=0.0, fifo_uram18=0.0, dwc_lut=0.0)
    flat, prev_output = [], None
    for stage, kind, r in blocks:
        flat.extend(_fifo_entries(kind, r, stage, prev_output))
        prev_output = block_output_name(stage, kind)
        g, costs = r.fifo_graph, r.fifo_costs
        fl = [dict(name=n, producer=g["fifos"][n]["producer"], consumer=g["fifos"][n]["consumer"], width_bits=int(g["fifos"][n]["bits"]), depth=int(g["fifos"][n]["depth"]),
                   max_occ=int(g["fifos"][n].get("max_occ", 0)), mem=c["mem"], depth_alloc=int(c["depth_alloc"]), lut=int(c["lut"]), bram18=int(c["bram18"]), uram18=int(c["uram"]))
              for n, c in costs.items() if g["fifos"][n]["depth"] > 2]
        rep[stage] = dict(kind=kind, verified=r.verification["ok"], steady_cyc_px=r.verification["steady_cyc_px"], T=r.params.get("T"), fifos=fl,
                          n_fifos_depth_le_2_removed_by_finn=len(costs) - len(fl),
                          dwcs=[dict(edge=d.edge, in_width=d.in_width, out_width=d.out_width, lut=d.lut) for d in r.dwcs], warnings=list(r.warnings))
        for k in ("fifo_lut", "fifo_bram18", "dwc_lut"):
            tot[k] += r.totals[k]
        tot["fifo_uram18"] += r.totals["fifo_uram"]
    result["block_verification"] = rep          # per-block summary
    result["intra_block_fifos"] = flat          # the FLAT per-FIFO list the FINN bridge reads, built from the verified blocks (verify_blocks runs them in parallel, one per distinct shape)
    dg = result["_diagnostics"]
    dg["intra_block_fifo_totals"] = tot
    dg["intra_block_fifo_bram18"] = sum(f["mem_bram18"] for f in flat)
    dg["intra_block_fifo_warnings"] = []
    dg["total_bram18k_with_all_fifos"] = dg["total_bram18k_calibrated"] + dg["inter_block_fifo_bram18"] + dg["intra_block_fifo_bram18"]


def finalize_reports(result):
    fixed = result["_diagnostics"]["fixed_cycle_nodes"]
    result["_diagnostics"]["branch_imbalance"] = compute_branch_imbalance_report(result)
    result["_diagnostics"]["chain_rate_imbalance"] = compute_chain_rate_imbalance_report(result, fixed)


# ---------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config_12_dense_relu_nearest_upsample_256")
    ap.add_argument("--bits", type=int, default=6, help="uniform weight = activation bits")
    ap.add_argument("--fps", type=float, default=250.0, help="throughput target: every node's cycles per frame <= clock / fps")
    ap.add_argument("--clock-mhz", type=float, default=100.0)
    ap.add_argument("--max-latency-ms", type=float, default=None,
                    help="latency cap = time to the first output pixel (sum of the blocks' first-in -> first-out latencies / clock): the per-node budget is lowered (bisection) until it holds")
    ap.add_argument("--ratchet-pct", type=float, default=4.0, help="allowance of the downstream-faster ratchet in PERCENT (the MILP's --ratchet-pct default); a block may be at most this much slower than the one before")
    ap.add_argument("--ratchet-floor", type=float, default=0.6, help="the ratchet never tightens a block below this fraction of the per-node budget")
    ap.add_argument("--mvau-wwidth-max", type=int, default=None,
                    help="cap on weight_bits * SIMD of every MVAU (FINN's mvau_wwidth_max; the MILP's flag of the same name). Use the same value as the MILP arms and the FINN auto-fold control")
    ap.add_argument("--no-ratchet", action="store_true", help="every block gets the full per-node budget (stages matched to it, no downstream-faster rule)")
    ap.add_argument("--tag", default="final", help="file tag: layer_bits_folding_<tag>.json / layer_bits_SITES_<tag>.json (S12 artifacts use 'final')")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--no-sites", action="store_true", help="skip the expand_layer_bits.py step")
    ap.add_argument("--inter-fifo", choices=("fixed", "sim", "standard"), default="fixed",
                    help="inter-block FIFO depths: 'fixed' = --inter-fifo-depth words each (default); 'sim' = smallest depth per block pair that "
                         "sustains --fifo-target-fps (net_fifo.py); 'standard' = one BRAM18 each")
    ap.add_argument("--inter-fifo-depth", type=int, default=2, help="depth of every inter-block FIFO for --inter-fifo fixed (2 = what FINN's RemoveShallowFIFOs deletes)")
    ap.add_argument("--fifo-target-fps", type=float, default=None, help="rate the inter-block FIFOs must sustain (default: --fps, i.e. the global requirement)")
    ap.add_argument("--fifo-slack", type=float, default=0.0, help="accepted last-frame period overshoot of the pair / whole-net runs over the target")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="parallel block verifications / pair simulations")
    ap.add_argument("--whole-net-check", action="store_true", help="run a closing whole-net simulation of the chained blocks with the inter-block depths (slow, ~10 min; off by default)")
    ap.add_argument("--no-verify", action="store_true", help="skip block verification, intra-block FIFO report, FIFO sizing and the ONNX picture (implies --inter-fifo standard)")
    ap.add_argument("--no-onnx", action="store_true", help="skip the whole-net ONNX picture")
    a = ap.parse_args()

    import bottleneck as _bn
    _bn.WWIDTH_MAX = a.mvau_wwidth_max
    finn_milp.load_config(a.config)
    finn_milp.CANDIDATE_BITS = tuple(sorted(set(finn_milp.CANDIDATE_BITS) | {a.bits}))
    model, geoms, extras, _pred, dmap, kinds = build_model_and_graph()
    ctx = (geoms, extras, {g.name: g for g in geoms}, {n.geom.name: n for n in extras}, dmap, kinds)
    F = int(a.clock_mhz * 1e6 / a.fps)
    cap = a.max_latency_ms * a.clock_mhz * 1e3 if a.max_latency_ms else None
    print(f"throughput target {a.fps:g} fps @ {a.clock_mhz:g} MHz -> every node <= {F} cycles/frame"
          + (f"; latency cap {a.max_latency_ms:g} ms -> first output pixel within {cap:.0f} cycles" if cap else "")
          + f"; INT{a.bits} uniform; {len(geoms)} layers, {len(extras)} extra nodes", flush=True)

    result = assemble(a, ctx, F, keep_blocks=True)
    final_F_top = F
    if cap is not None and result["_diagnostics"]["latency_first_out_cycles"] > cap:
        lo, hi = 81_920, F       # hi violates the cap; lo = the init maxpool floor (not foldable). Find the largest per-node budget in [lo, hi) that holds
        while True:
            try:
                best = assemble(a, ctx, lo, keep_blocks=True)
                break
            except ValueError:   # some block's own floor (maxpool, UpsampleNearestNeighbour, FMPadding_Pixel) is above lo
                lo = int(lo * 1.05)
        if best["_diagnostics"]["latency_first_out_cycles"] > cap:
            raise SystemExit(f"latency cap {a.max_latency_ms} ms infeasible: even at a per-node budget of {lo} cycles the first output pixel takes "
                             f"{best['_diagnostics']['latency_ms']:.1f} ms")
        while hi - lo > max(500, lo // 50):
            mid = (lo + hi) // 2
            trial = assemble(a, ctx, mid, keep_blocks=True)
            ok = trial["_diagnostics"]["latency_first_out_cycles"] <= cap
            print(f"  per-node budget {mid:7d}: first-out {trial['_diagnostics']['latency_ms']:6.1f} ms  {'ok' if ok else 'over'}", flush=True)
            if ok:
                lo, best = mid, trial
            else:
                hi = mid
        result = best
        final_F_top = lo
    finalize_reports(result)
    d = result["_diagnostics"]
    blocks = result.pop("_blocks")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_info = None
    if not a.no_verify:
        blocks = verify_blocks(blocks, a.workers)
        intra_block_report(blocks, result)
        ifaces = None
        if a.inter_fifo in ("fixed", "sim"):
            ftarget = int(a.clock_mhz * 1e6 / (a.fifo_target_fps or a.fps))
            ifaces = size_inter_block_fifos(result, blocks, a, ftarget)
        if not a.no_onnx:
            from net_onnx import export_net_onnx
            if ifaces is None:
                by_src = {f["src_block"]: f for f in result["inter_block_fifos"]}
                ifaces = [dict(depth=by_src[blocks[i][0]]["depth"], width_bits=by_src[blocks[i][0]]["width_bits"]) for i in range(len(blocks) - 1)]
            onnx_path = a.out_dir / f"enet_dataflow_{a.tag}.onnx"
            summary = (f"{a.config} INT{a.bits} uniform, {a.fps:g} fps @ {a.clock_mhz:g} MHz | {len(blocks)} blocks | LUT {d['total_lut_calibrated']:.0f} "
                       f"BRAM18 {d['total_bram18k_calibrated']:.0f} DSP {d['total_dsp']:.0f} | inter-block FIFOs: {d['inter_block_fifo_bram18']} BRAM18, "
                       f"{d.get('inter_block_fifo_lut', 0)} LUT | intra-block FIFO depth<=2 removed (RemoveShallowFIFOs)")
            onnx_info = export_net_onnx(blocks, ifaces, str(onnx_path), summary)
            onnx_info["path"] = str(onnx_path)
    out = a.out_dir / f"layer_bits_folding_{a.tag}.json"
    out.write_text(json.dumps(result, indent=2))
    prof = d["analytical"]["block_profile"]
    with open(a.out_dir / f"block_profile_{a.tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prof[0]))
        w.writeheader()
        w.writerows(prof)
    (a.out_dir / "run_args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2))
    if not a.no_sites:
        write_sites(a.config, out, a.out_dir / f"layer_bits_SITES_{a.tag}.json")
    cri = d["chain_rate_imbalance"]
    for pr in prof:
        print(f"  {pr['block']:12s} {pr['kind']:5s} budget {pr['budget']:7d}  slowest foldable {pr['slowest_foldable']:7d}")
    print(f"\nper-node budget used {d['per_node_budget_used']}  LUT {d['total_lut_calibrated']:.0f} ({d['lut_pct_of_budget']:.1f}%)  "
          f"BRAM18 {d['total_bram18k_calibrated']:.0f} + {d['inter_block_fifo_bram18']} inter-block FIFO  DSP {d['total_dsp']:.0f} ({d['dsp_pct_of_budget']:.1f}%)")
    print(f"bottleneck {d['bottleneck_node']} {d['bottleneck_cycles']} cyc -> {d['fps']:.1f} fps @ {a.clock_mhz:g} MHz; latency to first output pixel {d['latency_ms']:.2f} ms (sum of node cycles {d['sum_of_node_cycles_ms']:.1f} ms, not a latency); "
          f"pipeline-fill estimate {d['analytical']['sum_of_block_first_out_latencies_cycles'] / (a.clock_mhz * 1e3):.2f} ms")
    print(f"DSR (cycles per output element): max {cri.get('max_ratio')} median {cri.get('median_ratio')};  MVAU cycle mismatches vs MILP cost model: "
          f"{len(d['analytical']['mvau_cycle_mismatches_vs_milp_cost_model'])}")
    print(f"intra-block FIFOs: {len(result['intra_block_fifos'])} edges, {d['intra_block_fifo_bram18']} BRAM18; total BRAM18 with ALL FIFOs (inter + intra block): {d['total_bram18k_with_all_fifos']:.0f}")
    if "fifo_sizing" in d:
        fs = d["fifo_sizing"]
        depths = sorted(f["depth"] for f in result["inter_block_fifos"])
        print(f"inter-block FIFOs ({'fixed' if 'fixed' in fs['method'] else 'pair simulation'}, target {fs['target_cycles_per_frame']} cyc/frame): depths min {depths[0]} median {depths[len(depths) // 2]} max {depths[-1]}; "
              f"{d['inter_block_fifo_bram18']} BRAM18 + {d['inter_block_fifo_lut']} LUT + {d['inter_block_fifo_uram18']} URAM; whole-net check: {fs['whole_net']}")
    if onnx_info:
        print(f"wrote {onnx_info['path']}: {onnx_info['nodes']} nodes ({onnx_info['fifos']} FIFOs, {onnx_info['dwcs']} DWCs)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
