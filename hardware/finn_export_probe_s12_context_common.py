"""Shared core for the S12 dense-vs-separable context-block probe pair (see
finn_export_probe_s12_context_dense_int4.py / _separable_int4.py, the two
thin per-variant entry points that import this module).

Two standalone 2-bottleneck networks, identical in every respect except
`separable_dilated`, isolating that one architectural difference for a real
FINN OOC-synthesis comparison -- see the plan this implements:
C:\\Users\\win32\\.claude\\plans\\i-am-trying-to-jazzy-llama.md.

Topology: 1x1 QuantConv2d stem (IN_CHANNELS=1 -> CHANNELS=32, BN, QuantReLU --
produces a properly INT4-quantized activation stream for the first real probed
block, same convention as finn_export_probe_upsample_nearest_depthwise_int8.py's
own stem) -> QuantRegularBottleneck(dilation=2) -> QuantRegularBottleneck(dilation=4),
both real S12 context-stage geometry (channels=32, internal_ratio=4 ->
internal_channels=8, kernel_size=3, padding=dilation), uniform INT4
(weight_bit_width=act_bit_width=4), dummy (randomly-initialized) weights --
this is a pure resource/synthesis probe, not a trained model.

`QuantRegularBottleneck` is imported directly from the real
enet/nnunetv2/nets/QuantENet.py (confirmed importable + brevitas/qonnx/onnx
all present in the `lightmunet_dev` container this session already uses) --
no hand-duplicated architecture code, unlike the older upsample/depthwise
probe (which avoided nnunetv2 for a since-resolved environment reason, see
that script's own docstring).

Raw (UNDERATED) analytical LUT/BRAM estimate is computed BEFORE any ONNX
export, via a small ENet-agnostic forward-hook geometry capture (same
technique as compression/hawq/finn_block_costs.py's own dump_block_layer_geometry
hook(), just without needing enumerate_blocks/path_to_block_map's ENet-shaped
12-attribute contract, since this is a bare 2-block module) feeding directly
into finn_cost_model.layer_cost_pe_simd(force_dsp=True, no calibration
applied) -- PE=SIMD=1 for every layer, matching the folding this build's own
FINN config will also force via target_fps=None (see
finn_build_probe_s12_context_int4.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(REPO_ROOT / "compression" / "hawq"))

from nnunetv2.nets.QuantENet import QuantRegularBottleneck, _quant_conv2d, _quant_act  # noqa: E402
from finn_cost_model import LayerGeometry, layer_cost_pe_simd, RAM_STYLE_BLOCK  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "finn_exports"

# -- Real S12 context-stage geometry (compression/hawq/config_12_separable_dense_relu.py's
# CHANNELS=(4,16,32,16,4) -- stage2/3 context width -- and QuantRegularBottleneck's
# own internal_ratio=4 default) --------------------------------------------
IN_CHANNELS = 1
CHANNELS = 32
INTERNAL_RATIO = 4  # -> internal_channels = 8, matches real S12
KERNEL_SIZE = 3
DILATIONS = (2, 4)  # first two rungs of S12's real dense_dilation schedule
WEIGHT_BIT_WIDTH = 4
ACT_BIT_WIDTH = 4
INPUT_HW = (32, 32)
PE = SIMD = 1  # "same folding" -- fully serial, matches this session's earlier experiments


def _pair(v) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else tuple(v)


class ProbeNet(nn.Module):
    """1x1 stem -> 2 stacked QuantRegularBottleneck context blocks, only
    `separable_dilated` varies between the dense/separable probes."""

    def __init__(self, separable_dilated: bool):
        super().__init__()
        self.stem = nn.Sequential(
            _quant_conv2d(IN_CHANNELS, CHANNELS, WEIGHT_BIT_WIDTH, kernel_size=1),
            nn.BatchNorm2d(CHANNELS),
            _quant_act(ACT_BIT_WIDTH),
        )
        self.bottlenecks = nn.ModuleList([
            QuantRegularBottleneck(
                channels=CHANNELS, weight_bit_width=WEIGHT_BIT_WIDTH, act_bit_width=ACT_BIT_WIDTH,
                internal_ratio=INTERNAL_RATIO, kernel_size=KERNEL_SIZE, padding=dilation, dilation=dilation,
                separable_dilated=separable_dilated,
            )
            for dilation in DILATIONS
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for bneck in self.bottlenecks:
            x = bneck(x)
        return x


def dump_probe_geometry(model: nn.Module, input_hw: tuple[int, int]) -> list[LayerGeometry]:
    """ENet-agnostic version of finn_block_costs.py's dump_block_layer_geometry
    hook -- tags each Conv2d by which top-level child (stem/bottlenecks.N) it's
    under, without needing enumerate_blocks/path_to_block_map's ENet-shaped
    12-attribute contract (a bare ProbeNet doesn't expose those)."""
    geometries: list[LayerGeometry] = []

    def stage_for(name: str) -> str:
        return name.split(".")[0] if not name.startswith("bottlenecks.") else ".".join(name.split(".")[:2])

    def make_hook(name: str, op_type: str):
        def hook(module, inputs, output):
            x = inputs[0]
            if isinstance(output, tuple):
                output = output[0]
            kh, kw = _pair(module.kernel_size)
            sh, sw = _pair(module.stride)
            dh, dw = _pair(getattr(module, "dilation", 1))
            geometries.append(LayerGeometry(
                op_type=op_type, name=name, stage=stage_for(name),
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


def raw_estimate(geometries: list[LayerGeometry]) -> dict:
    """Raw (UNDERATED -- no calibrated_lut/calibrated_bram18k applied)
    per-layer and summed LUT/BRAM/cycles, force_dsp=True, PE=SIMD=1 for every
    layer."""
    per_layer = {}
    total_lut = total_bram18 = total_uram18 = 0.0
    stem_lut = stem_bram18 = 0.0
    for g in geometries:
        r = layer_cost_pe_simd(g, WEIGHT_BIT_WIDTH, ACT_BIT_WIDTH, PE, SIMD, ram_style=RAM_STYLE_BLOCK, force_dsp=True)
        per_layer[g.name] = {
            "stage": g.stage, "cin": g.cin, "cout": g.cout, "hin": g.hin, "win": g.win,
            "kh": g.kh, "kw": g.kw, "dh": g.dh, "dw": g.dw, "groups": g.groups,
            **r,
        }
        total_lut += r["total_lut"]
        total_bram18 += r["swu_bram18"] + r["wm_bram18"]
        total_uram18 += r["wm_uram18"]
        if g.stage == "stem":
            stem_lut += r["total_lut"]
            stem_bram18 += r["swu_bram18"] + r["wm_bram18"]
    return {
        "per_layer": per_layer,
        "total_lut": total_lut, "total_bram18": total_bram18, "total_uram18": total_uram18,
        "bottlenecks_only_lut": total_lut - stem_lut, "bottlenecks_only_bram18": total_bram18 - stem_bram18,
        "n_layers": len(geometries),
    }


def build_and_estimate(separable_dilated: bool) -> tuple[ProbeNet, dict]:
    torch.manual_seed(0)  # dummy weights, but reproducible across runs
    model = ProbeNet(separable_dilated)
    geometries = dump_probe_geometry(model, INPUT_HW)
    est = raw_estimate(geometries)
    return model, est


def export_onnx(model: nn.Module, name: str) -> Path:
    """Same export helper as hardware/finn_enet_prod_export.py's export_model
    (reimplemented inline here to keep this probe self-contained and avoid
    that file's own nnunetv2-independent-import assumption)."""
    from brevitas.export import export_qonnx
    from qonnx.util.cleanup import cleanup as qonnx_cleanup
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.datatype import DataType
    import onnx

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.onnx"

    model.cpu().eval()
    dummy = torch.randn(1, IN_CHANNELS, *INPUT_HW)
    export_qonnx(model, export_path=str(out_path), input_t=dummy)
    qonnx_cleanup(str(out_path), out_file=str(out_path))

    qm = ModelWrapper(str(out_path))
    qm.set_tensor_datatype(qm.graph.input[0].name, DataType["INT8"])
    # Final activation is _quant_act -> QuantReLU/Uint8ActPerTensorFloat (unsigned,
    # non-narrow, see QuantENet._quant_act's own docstring) -- annotating the graph
    # output as signed INT8 here caused FINN's convert_to_hw step to fail with
    # a similar "Signed output requires actval < 0" (the real output never
    # goes negative, so the emitted thresholds have no negative actval to satisfy
    # a signed-output annotation).
    qm.set_tensor_datatype(qm.graph.output[0].name, DataType[f"UINT{ACT_BIT_WIDTH}"])
    qm.save(str(out_path))

    loaded = onnx.load(str(out_path))
    assert len(loaded.graph.node) > 0, "exported model has no nodes"
    ops: dict[str, int] = {}
    for n in loaded.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"  {name}: {len(loaded.graph.node)} nodes -- {dict(sorted(ops.items()))}")
    print(f"  Saved: {out_path}")
    return out_path


def run(name: str, separable_dilated: bool) -> dict:
    """Full per-variant pipeline: build -> raw estimate -> export ONNX ->
    return a summary dict (for the combined comparison JSON)."""
    print(f"=== {name} (separable_dilated={separable_dilated}) ===")
    model, est = build_and_estimate(separable_dilated)
    print(f"  layers: {est['n_layers']}  raw_total_lut={est['total_lut']:.1f}  "
          f"raw_total_bram18={est['total_bram18']:.1f}  raw_total_uram18={est['total_uram18']:.1f}")
    onnx_path = export_onnx(model, name)
    return {
        "separable_dilated": separable_dilated,
        "onnx_path": str(onnx_path),
        "architecture": {
            "in_channels": IN_CHANNELS, "channels": CHANNELS, "internal_ratio": INTERNAL_RATIO,
            "internal_channels": CHANNELS // INTERNAL_RATIO, "kernel_size": KERNEL_SIZE,
            "dilations": list(DILATIONS), "weight_bit_width": WEIGHT_BIT_WIDTH, "act_bit_width": ACT_BIT_WIDTH,
            "pe": PE, "simd": SIMD, "input_hw": list(INPUT_HW), "force_dsp": True,
        },
        "raw_estimate": est,
    }


def write_combined_summary(dense_result: dict, separable_result: dict) -> Path:
    combined = {
        "dense": dense_result,
        "separable": separable_result,
        "ratio_dense_over_separable": {
            "raw_total_lut": dense_result["raw_estimate"]["total_lut"] / separable_result["raw_estimate"]["total_lut"],
            "raw_total_bram18": dense_result["raw_estimate"]["total_bram18"] / separable_result["raw_estimate"]["total_bram18"],
            "bottlenecks_only_lut": dense_result["raw_estimate"]["bottlenecks_only_lut"] / separable_result["raw_estimate"]["bottlenecks_only_lut"],
        },
        "note": (
            "raw = UNDERATED analytical estimate (finn_cost_model.layer_cost_pe_simd, "
            "force_dsp=True, PE=SIMD=1, no calibrated_lut/calibrated_bram18k applied). "
            "Compare this ratio against real FINN build reports once "
            "finn_build_probe_s12_context_int4.py has been run through OOC synthesis."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "probe_s12_context_int4_analytical_estimate.json"
    out_path.write_text(json.dumps(combined, indent=2))
    print(f"\nWrote combined summary: {out_path}")
    return out_path
