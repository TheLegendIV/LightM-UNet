"""Probe network for settling the hls_lut_noact0/hls_dsp_noact0 ILP-variant
argument (see MILP/finn_milp.py's VARIANT_HLS_LUT_NOACT0/
VARIANT_HLS_DSP_NOACT0 and the plan this was built from,
C:\\Users\\win32\\.claude\\plans\\the-current-ilp-inherited-lynx.md): a real
Vivado OOC-synthesis probe isolating (a) the fused-activation (noActivation=0)
LUT/BRAM effect alone, and (b) the resType=dsp-vs-lut multiplier effect alone,
at FIXED noActivation=0 -- neither is isolated by any real data in this repo
today (hardware/probes/mvau_lut_calibration_dataset_s12_context_dense_int6_
pemh_simd1.csv is 100% noActivation=1/resType=dsp; hardware/datasets/
mvau_lut_calibration_dataset.csv is 100% resType=dsp, zero variance on that
axis, so it says nothing about resType=lut).

WHY A PLAIN SEQUENTIAL NETWORK, NOT QuantRegularBottleneck (deliberate
deviation, confirmed with the user 2026-09-17): FINN's own
InferQuantizedMatrixVectorActivation (convert_to_hw_layers.py) fuses a
MatMul's trailing activation into noActivation=0 PURELY STRUCTURALLY -- iff
model.find_consumer(mm_output) is a bare MultiThreshold node, no fork/
residual-add in between (confirmed via direct source read, lines ~2016-2077
of that file). QuantRegularBottleneck's own `expand` sub-block structurally
CANNOT fuse (its output feeds `residual_add`, an Add node, not a
MultiThreshold -- `out_act` only applies AFTER the join, see QuantENet.py's
QuantRegularBottleneck.forward()); real data confirms ALL 7 layers of the
existing bottleneck-based s12_context probe landed noActivation=1 (uniformly
MVAU_rtl, which structurally REQUIRES noActivation=1), including `reduce`/
`conv_bn_act`, whose own trailing activation IS immediately-following in the
un-joined case -- suggesting some other streamlining-order effect suppresses
fusion there too that could not be fully resolved by static source reading
alone (would need to inspect the ACTUAL intermediate model this repo's own
FINN container produces, which this environment cannot run). Removing the
residual join entirely -- a plain sequential chain, each conv-owning
sub-block ending in its OWN dedicated activation with nothing else in
between -- makes fusion structurally CERTAIN for every layer, not something
hoped for. This trades network "realism" for experimental control, matching
this repo's own established probe convention (every probe here is a small
synthetic geometry, never a real deployment model, see
finn_export_probe_s12_context_common.py's own docstring).

Topology (2026-09-17 SIMPLIFIED to a single reduce/conv/expand group,
down from 2 -- fewer nodes means faster synthesis and unambiguous per-node
attribution, and this probe's whole point is isolating axes, not matching a
full real network): input already at channels=32 (no separate 1-channel
stem -- the network's own graph input IS the 32-channel tensor), internal_
ratio=4 -> internal_channels=8, kernel_size=3, dilation=16 (worst case: the
largest dilation used anywhere in the real S12 dense schedule, maximizing
the SWU/line-buffer footprint alongside the already-worst-case W8A8 bits),
chained sequentially with NO residual add:
    reduce: 1x1 Conv(32->8)  -> BN -> act(8)
    conv:   3x3 Conv(8->8, dilation=16, padding=16) -> BN -> act(8)
    expand: 1x1 Conv(8->32) -> BN -> act(32)   <- act() added here (the ONE
                                                   deviation from
                                                   QuantRegularBottleneck's
                                                   own `expand`, which has NO
                                                   activation of its own --
                                                   quantization normally
                                                   happens in out_act AFTER
                                                   the residual join, which no
                                                   longer exists here)
3 conv layers total (down from 7). W8A8 (2026-09-17: switched from W6A6 to
the worst-case/max bit-width this repo's ILP actually searches over,
CANDIDATE_BITS' upper end -- LUT usage scales with bit-width, so INT8 is the
conservative choice for a probe meant to bound real risk, not just
characterize a convenient reference point).

CANONICAL WEIGHTS (2026-09-17): the matching reduce/conv/expand conv
weights, AND the export-tracing input tensor, are loaded from
_probe_canonical_weights.py -- shared verbatim with finn_export_probe_
noact1_single_int8.py -- instead of left at each script's own from-scratch
random init. torch.manual_seed(0) alone does NOT give the two networks the
same actual weight values (different module graphs consume the global RNG
stream differently during construction, see that shared module's own
docstring for the full reasoning, including why this matters for real FINN
RTL-eligibility checks, not just cosmetically).

Usage:
    python3 finn_export_probe_lutmult_noact0.py
Then, inside the FINN container, run finn_build_probe_lutmult_noact0.py once
per (--res-type, --fold) combination (2x3 = 6 real Vivado OOC synthesis
builds) -- see that script's own docstring. For the noActivation=1 side of
the matrix (rtl-auto and hls-forced, same resType/fold axes), reuse the
EXISTING probe_s12_context_dense_int8.onnx (already exported, bottleneck
topology reliably lands noActivation=1) with finn_build_probe_s12_context_
noact1_int8.py -- no second export script needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # hardware/probes/mvau_variant_matrix/<this file>.py -> repo root
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(REPO_ROOT / "MILP"))

from nnunetv2.nets.QuantENet import _quant_conv2d, _quant_act  # noqa: E402
from finn_cost_model import LayerGeometry, layer_cost_pe_simd_auto_ram  # noqa: E402
from _probe_canonical_weights import canonical_dummy_input, load_canonical_weights_  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "finn_exports"
MODEL_NAME = "probe_lutmult_noact0_int8"

IN_CHANNELS = 32  # network's own graph input IS the 32-channel tensor -- no separate 1-channel stem
CHANNELS = 32
INTERNAL_RATIO = 4  # -> internal_channels = 8, matches real S12 and the existing bottleneck probe
KERNEL_SIZE = 3
DILATION = 16  # worst case: largest dilation in the real S12 dense schedule
BIT_WIDTH = 8  # W8A8, worst-case bit-width -- see module docstring
INPUT_HW = (32, 32)


def _pair(v) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else tuple(v)


class _ConvBnAct(nn.Sequential):
    """One conv-owning sub-block: conv -> BN -> act, ALWAYS ending in its own
    dedicated activation (unlike QuantRegularBottleneck's `expand`, which has
    none) -- see module docstring for why every block needs one here."""

    def __init__(self, in_ch: int, out_ch: int, bit_width: int, **conv_kwargs):
        super().__init__(
            _quant_conv2d(in_ch, out_ch, bit_width, **conv_kwargs),
            nn.BatchNorm2d(out_ch),
            _quant_act(bit_width),
        )


class SequentialProbeNet(nn.Module):
    """Single QuantRegularBottleneck-shaped reduce/conv/expand group
    (dilation=16), residual join REMOVED -- see module docstring. Input is
    already at CHANNELS=32 -- no separate stem."""

    def __init__(self, bit_width: int = BIT_WIDTH):
        super().__init__()
        internal_channels = max(1, CHANNELS // INTERNAL_RATIO)
        self.blocks = nn.Sequential(
            _ConvBnAct(CHANNELS, internal_channels, bit_width, kernel_size=1),  # reduce
            _ConvBnAct(internal_channels, internal_channels, bit_width,
                       kernel_size=KERNEL_SIZE, padding=DILATION, dilation=DILATION),  # conv
            _ConvBnAct(internal_channels, CHANNELS, bit_width, kernel_size=1),  # expand
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

    def conv_modules(self) -> dict[str, nn.Conv2d]:
        """{"reduce"/"conv"/"expand": the actual nn.Conv2d} -- for
        _probe_canonical_weights.load_canonical_weights_."""
        return {"reduce": self.blocks[0][0], "conv": self.blocks[1][0], "expand": self.blocks[2][0]}


def dump_probe_geometry(model: nn.Module, input_hw: tuple[int, int]) -> list[LayerGeometry]:
    """Same forward-hook technique as finn_export_probe_s12_context_common.py's
    dump_probe_geometry -- tags each Conv2d by its dotted module path."""
    geometries: list[LayerGeometry] = []

    def make_hook(name: str, op_type: str):
        def hook(module, inputs, output):
            x = inputs[0]
            kh, kw = _pair(module.kernel_size)
            sh, sw = _pair(module.stride)
            dh, dw = _pair(getattr(module, "dilation", 1))
            geometries.append(LayerGeometry(
                op_type=op_type, name=name, stage=name.split(".")[0],
                cin=x.shape[1], hin=x.shape[2], win=x.shape[3],
                cout=output.shape[1], hout=output.shape[2], wout=output.shape[3],
                kh=kh, kw=kw, sh=sh, sw=sw, dh=dh, dw=dw,
                groups=getattr(module, "groups", 1),
            ))
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook(name, "Conv2d")))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, IN_CHANNELS, *input_hw))
    for h in handles:
        h.remove()
    return geometries


def raw_estimate(geometries: list[LayerGeometry], bit_width: int = BIT_WIDTH) -> dict:
    """Raw (UNDERATED) per-layer LUT/BRAM/DSP at PE=SIMD=1, for BOTH resType
    arms this probe is meant to check -- a pre-synthesis reference point to
    compare the eventual real Vivado numbers against, same convention as
    finn_export_probe_s12_context_common.py's own raw_estimate but extended
    with no_activation=False (this probe's whole point) and both resType
    choices (force_dsp True/False) side by side."""
    per_layer = {}
    totals = {"dsp": {"lut": 0.0, "bram18": 0.0, "dsp_count": 0.0}, "lut": {"lut": 0.0, "bram18": 0.0, "dsp_count": 0.0}}
    for g in geometries:
        per_layer[g.name] = {"stage": g.stage, "cin": g.cin, "cout": g.cout, "kh": g.kh, "kw": g.kw, "dh": g.dh}
        for res_type, force_dsp in (("dsp", True), ("lut", False)):
            # impl_style="hls" is REQUIRED here -- conv_cost_pe_simd's use_dsp
            # is `force_dsp or impl_style==RTL`, and impl_style defaults to
            # RTL, so omitting this collapses BOTH regimes to DSP regardless
            # of force_dsp (caught 2026-09-17: dsp/lut raw estimates were
            # printing identical total_dsp until this was added).
            r = layer_cost_pe_simd_auto_ram(g, bit_width, bit_width, pe=1, simd=1,
                                             impl_style="hls", force_dsp=force_dsp, no_activation=False)
            per_layer[g.name][res_type] = r
            totals[res_type]["lut"] += r["total_lut"]
            totals[res_type]["bram18"] += r["swu_bram18"] + r["wm_bram18"]
            totals[res_type]["dsp_count"] += r["total_dsp"]
    return {"per_layer": per_layer, "totals": totals, "n_layers": len(geometries)}


def export_onnx(model: nn.Module, name: str, act_bit_width: int = BIT_WIDTH) -> Path:
    """Same export helper as finn_export_probe_s12_context_common.py's
    export_onnx (reimplemented inline to keep this probe self-contained)."""
    from brevitas.export import export_qonnx
    from qonnx.util.cleanup import cleanup as qonnx_cleanup
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.datatype import DataType
    import onnx

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.onnx"

    model.cpu().eval()
    dummy = canonical_dummy_input(IN_CHANNELS, INPUT_HW)
    export_qonnx(model, export_path=str(out_path), input_t=dummy)
    qonnx_cleanup(str(out_path), out_file=str(out_path))

    qm = ModelWrapper(str(out_path))
    qm.set_tensor_datatype(qm.graph.input[0].name, DataType["INT8"])
    # _quant_act is QuantReLU/Uint8ActPerTensorFloat (unsigned, non-narrow) --
    # same convention as the existing s12_context probe's export_onnx.
    qm.set_tensor_datatype(qm.graph.output[0].name, DataType[f"UINT{act_bit_width}"])
    qm.save(str(out_path))

    loaded = onnx.load(str(out_path))
    assert len(loaded.graph.node) > 0, "exported model has no nodes"
    ops: dict[str, int] = {}
    for n in loaded.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"  {name}: {len(loaded.graph.node)} nodes -- {dict(sorted(ops.items()))}")
    print(f"  Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    print(f"=== {MODEL_NAME} (W{BIT_WIDTH}A{BIT_WIDTH}, sequential, no residual) ===")
    torch.manual_seed(0)  # BN/quantizer init reproducible across runs of THIS script
    model = SequentialProbeNet(BIT_WIDTH)
    internal_channels = CHANNELS // INTERNAL_RATIO
    load_canonical_weights_(model.conv_modules(), CHANNELS, internal_channels, KERNEL_SIZE)
    print("  Loaded canonical (shared with finn_export_probe_noact1_single_int8.py) conv weights.")
    geometries = dump_probe_geometry(model, INPUT_HW)
    est = raw_estimate(geometries, BIT_WIDTH)
    print(f"  layers: {est['n_layers']}")
    for res_type in ("dsp", "lut"):
        t = est["totals"][res_type]
        print(f"  raw (no_activation=False, resType={res_type}): "
              f"total_lut={t['lut']:.1f}  total_bram18={t['bram18']:.1f}  total_dsp={t['dsp_count']:.1f}")

    onnx_path = export_onnx(model, MODEL_NAME, BIT_WIDTH)

    summary = {
        "model_name": MODEL_NAME,
        "onnx_path": str(onnx_path),
        "architecture": {
            "in_channels": IN_CHANNELS, "channels": CHANNELS, "internal_ratio": INTERNAL_RATIO,
            "internal_channels": CHANNELS // INTERNAL_RATIO, "kernel_size": KERNEL_SIZE,
            "dilation": DILATION, "bit_width": BIT_WIDTH, "input_hw": list(INPUT_HW),
            "topology": "sequential, no residual (see module docstring)",
        },
        "raw_estimate": est,
        "note": (
            "raw = UNDERATED analytical estimate (finn_cost_model.layer_cost_pe_simd_auto_ram, "
            "no_activation=False, PE=SIMD=1), for BOTH resType arms. Compare against real FINN "
            "OOC-synthesis reports once finn_build_probe_lutmult_noact0.py has been run for each "
            "(--res-type, --fold) combination."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"{MODEL_NAME}_analytical_estimate.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote analytical-estimate summary: {summary_path}")
