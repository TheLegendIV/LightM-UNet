"""noActivation=1 (UNFUSED) counterpart to finn_export_probe_lutmult_noact0.py
-- SAME geometry (single reduce/1x1 -> conv/3x3 dilation=16 -> expand/1x1
group, in/out channels=32, W8A8), but keeps QuantRegularBottleneck's
residual join, so the network structurally lands noActivation=1 on every
layer (real data already confirms this topology does, see finn_export_probe_
lutmult_noact0.py's own docstring for the full reasoning on why fusion is
purely graph-structural and this join blocks it). Feeds finn_build_probe_
s12_context_noact1_int8.py -- see the plan this was built from,
C:\\Users\\win32\\.claude\\plans\\the-current-ilp-inherited-lynx.md.

Deliberately matches finn_export_probe_lutmult_noact0.py's geometry exactly
(channels=32, internal_ratio=4 -> internal_channels=8, kernel_size=3,
dilation=16, in/out=32ch, no separate stem) so the noAct=0 vs noAct=1 arms
of the combination matrix differ ONLY in the axis under test, not in
geometry too -- confirmed with the user 2026-09-17. Does NOT replace
probe_s12_context_dense_int8.onnx (the older 2-block/dilations=(2,4)
export, kept untouched for whatever else already depends on it) -- this is
a new, separate, smaller network.

QuantRegularBottleneck is imported directly from the real
enet/nnunetv2/nets/QuantENet.py (same convention as finn_export_probe_
s12_context_common.py) -- no hand-duplicated architecture code, and no risk
of accidentally deviating from the SAME reduce/conv/expand/residual wiring
that already-real 7-layer dataset was built from.

CANONICAL WEIGHTS: the matching reduce/conv/expand conv weights, AND the
export-tracing input tensor, are loaded from _probe_canonical_weights.py --
shared verbatim with finn_export_probe_lutmult_noact0.py -- so the two
networks compare on identical weight VALUES, not just identical shapes/
seeds (torch.manual_seed(0) alone does NOT give two structurally different
module graphs the same actual weights -- see that shared module's own
docstring).

Usage:
    python3 finn_export_probe_noact1_single_int8.py
Then, inside the FINN container, run finn_build_probe_s12_context_noact1_
int8.py once per (--impl-style, --res-type, --fold) combination -- see that
script's own docstring (it already points at MODEL_NAME=
probe_noact1_single_d16_int8, update it if you keep the older 2-block
export around under a different name instead).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(REPO_ROOT / "compression" / "MILP"))

from nnunetv2.nets.QuantENet import QuantRegularBottleneck  # noqa: E402
from finn_cost_model import LayerGeometry, layer_cost_pe_simd_auto_ram  # noqa: E402
from _probe_canonical_weights import canonical_dummy_input, load_canonical_weights_  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "finn_exports"
MODEL_NAME = "probe_noact1_single_d16_int8"

IN_CHANNELS = 32  # network's own graph input IS the 32-channel tensor -- no separate stem, matches the noAct=0 sibling
CHANNELS = 32
INTERNAL_RATIO = 4  # -> internal_channels = 8
KERNEL_SIZE = 3
DILATION = 16  # worst case: largest dilation in the real S12 dense schedule, matches the noAct=0 sibling exactly
BIT_WIDTH = 8  # W8A8, worst-case bit-width, matches the noAct=0 sibling
INPUT_HW = (32, 32)


def _pair(v) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else tuple(v)


class SingleBottleneckProbeNet(nn.Module):
    """ONE real QuantRegularBottleneck (channels=32, dilation=16) -- residual
    join intact, so this structurally lands noActivation=1 on every layer
    (unlike this probe's noAct=0 sibling, which deliberately removes the
    join)."""

    def __init__(self, bit_width: int = BIT_WIDTH):
        super().__init__()
        self.block = QuantRegularBottleneck(
            channels=CHANNELS, weight_bit_width=bit_width, act_bit_width=bit_width,
            internal_ratio=INTERNAL_RATIO, kernel_size=KERNEL_SIZE, padding=DILATION, dilation=DILATION,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    def conv_modules(self) -> dict[str, nn.Conv2d]:
        """{"reduce"/"conv"/"expand": the actual nn.Conv2d} -- for
        _probe_canonical_weights.load_canonical_weights_. QuantRegularBottleneck's
        own `self.conv` is the bare QuantConv2d (not wrapped in a Sequential)
        for this plain non-asymmetric/non-dsc/non-separable_dilated path --
        see QuantENet.py's own constructor."""
        return {"reduce": self.block.reduce[0], "conv": self.block.conv, "expand": self.block.expand[0]}


def dump_probe_geometry(model: nn.Module, input_hw: tuple[int, int]) -> list[LayerGeometry]:
    """Same forward-hook technique as the noAct=0 sibling."""
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
    """Raw (UNDERATED) per-layer LUT/BRAM/DSP at PE=SIMD=1, no_activation=True
    (this network's structural regime), for the 3 real build targets this
    geometry will actually be run under: rtl (auto, always dsp), hls+dsp
    (forced), hls+lut (forced)."""
    per_layer = {}
    totals = {
        "rtl_dsp": {"lut": 0.0, "bram18": 0.0, "dsp_count": 0.0},
        "hls_dsp": {"lut": 0.0, "bram18": 0.0, "dsp_count": 0.0},
        "hls_lut": {"lut": 0.0, "bram18": 0.0, "dsp_count": 0.0},
    }
    regime_kwargs = {
        "rtl_dsp": {"impl_style": "rtl", "force_dsp": True},
        "hls_dsp": {"impl_style": "hls", "force_dsp": True},
        "hls_lut": {"impl_style": "hls", "force_dsp": False},
    }
    for g in geometries:
        per_layer[g.name] = {"stage": g.stage, "cin": g.cin, "cout": g.cout, "kh": g.kh, "kw": g.kw, "dh": g.dh}
        for regime, kw in regime_kwargs.items():
            r = layer_cost_pe_simd_auto_ram(g, bit_width, bit_width, pe=1, simd=1, no_activation=True, **kw)
            per_layer[g.name][regime] = r
            totals[regime]["lut"] += r["total_lut"]
            totals[regime]["bram18"] += r["swu_bram18"] + r["wm_bram18"] + r.get("thr_bram18", 0)
            totals[regime]["dsp_count"] += r["total_dsp"]
    return {"per_layer": per_layer, "totals": totals, "n_layers": len(geometries)}


def export_onnx(model: nn.Module, name: str, act_bit_width: int = BIT_WIDTH) -> Path:
    """Same export helper as the noAct=0 sibling and finn_export_probe_
    s12_context_common.py's own export_onnx."""
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
    print(f"=== {MODEL_NAME} (W{BIT_WIDTH}A{BIT_WIDTH}, single bottleneck, residual intact -> noActivation=1) ===")
    torch.manual_seed(0)  # BN/quantizer init reproducible across runs of THIS script
    model = SingleBottleneckProbeNet(BIT_WIDTH)
    internal_channels = CHANNELS // INTERNAL_RATIO
    load_canonical_weights_(model.conv_modules(), CHANNELS, internal_channels, KERNEL_SIZE)
    print("  Loaded canonical (shared with finn_export_probe_lutmult_noact0.py) conv weights.")
    geometries = dump_probe_geometry(model, INPUT_HW)
    est = raw_estimate(geometries, BIT_WIDTH)
    print(f"  layers: {est['n_layers']}")
    for regime, t in est["totals"].items():
        print(f"  raw (no_activation=True, {regime}): "
              f"total_lut={t['lut']:.1f}  total_bram18={t['bram18']:.1f}  total_dsp={t['dsp_count']:.1f}")

    onnx_path = export_onnx(model, MODEL_NAME, BIT_WIDTH)

    summary = {
        "model_name": MODEL_NAME,
        "onnx_path": str(onnx_path),
        "architecture": {
            "in_channels": IN_CHANNELS, "channels": CHANNELS, "internal_ratio": INTERNAL_RATIO,
            "internal_channels": CHANNELS // INTERNAL_RATIO, "kernel_size": KERNEL_SIZE,
            "dilation": DILATION, "bit_width": BIT_WIDTH, "input_hw": list(INPUT_HW),
            "topology": "single QuantRegularBottleneck, residual intact -> noActivation=1",
        },
        "raw_estimate": est,
        "note": (
            "raw = UNDERATED analytical estimate (finn_cost_model.layer_cost_pe_simd_auto_ram, "
            "no_activation=True, PE=SIMD=1), for all 3 real build regimes this geometry will run "
            "under. Compare against real FINN OOC-synthesis reports once finn_build_probe_s12_"
            "context_noact1_int8.py has been run for each (--impl-style, --res-type, --fold) "
            "combination."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"{MODEL_NAME}_analytical_estimate.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote analytical-estimate summary: {summary_path}")
