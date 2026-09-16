"""Export a MINIMAL standalone probe: nearest-neighbor upsample + depthwise
conv ("Approach 2" bilinear-upsampling decomposition discussed earlier),
smallest possible footprint (few channels, tiny spatial size), to verify it
lowers correctly all the way to a stitched IP in FINN (specifically that
`InferUpsample`/`UpsampleNearestNeighbour` + a depthwise `VVAU` chain
end-to-end, independent of and in parallel with the S12 partition-0 real
ZynqBuild).

Topology: 1x1 QuantConv2d (project 1->CHANNELS) -> BN -> QuantReLU ->
nn.Upsample(scale_factor=2, mode="nearest") -> depthwise (groups=CHANNELS)
3x3 QuantConv2d -> BN -> QuantReLU.

Fully self-contained (no dependency on the repo's enet/nnunetv2 package or
finn_enet_prod_export.py's export_model) -- the designated Brevitas-export
container for those (nice_wescoff) was lost the same way lucid_ptolemy was
(see memories/repo/finn_gotchas.md); this probe only needs brevitas+qonnx,
both already present inside the FINN container itself.

Usage (inside the FINN container):
    python3 finn_export_probe_upsample_nearest_depthwise_int8.py
"""
from pathlib import Path

import torch
from torch import nn
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat

OUT_DIR = Path(__file__).resolve().parent

CHANNELS = 4          # smallest non-trivial depthwise channel count
BIT_WIDTH = 8
IN_CHANNELS = 1
INPUT_HW = (8, 8)     # tiny -- fast HLS/Vivado IP-level synth
MODEL_NAME = "quant_probe_upsample_nearest_depthwise_int8"


def _quant_conv2d(in_ch: int, out_ch: int, bit_width: int, **kwargs) -> qnn.QuantConv2d:
    return qnn.QuantConv2d(
        in_ch, out_ch, bias=False,
        weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat,
        **kwargs,
    )


def _quant_act(bit_width: int) -> qnn.QuantReLU:
    # Uint8ActPerTensorFloat (signed=False, narrow_range=False) -- FINN's
    # dataflow backend rejects a QONNX-exported ReLU quantizer unless
    # unsigned+non-narrow (see enet/nnunetv2/nets/QuantENet.py's identical
    # helper for the full rationale).
    return qnn.QuantReLU(act_quant=Uint8ActPerTensorFloat, bit_width=bit_width)


def export_model(model: nn.Module, name: str, dummy: torch.Tensor) -> Path:
    """Export to cleaned QONNX, set INT8 datatypes (copied from
    finn_enet_prod_export.py's identical helper -- inlined to avoid that
    file's own nnunetv2 import-time dependency)."""
    from brevitas.export import export_qonnx
    from qonnx.util.cleanup import cleanup as qonnx_cleanup
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.datatype import DataType
    import onnx

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.onnx"

    model.cpu().eval()
    export_qonnx(model, export_path=str(out_path), input_t=dummy)
    qonnx_cleanup(str(out_path), out_file=str(out_path))

    qm = ModelWrapper(str(out_path))
    qm.set_tensor_datatype(qm.graph.input[0].name, DataType["INT8"])
    qm.set_tensor_datatype(qm.graph.output[0].name, DataType["INT8"])
    qm.save(str(out_path))

    loaded = onnx.load(str(out_path))
    assert len(loaded.graph.node) > 0, "exported model has no nodes"

    ops: dict = {}
    for n in qm.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"  {name}: {len(qm.graph.node)} nodes — {dict(sorted(ops.items()))}")
    print(f"  Saved: {out_path}")
    return out_path


class ProbeUpsampleDepthwise(nn.Module):
    """Minimal-footprint "Approach 2" bilinear-upsampling decomposition
    probe: nearest-neighbor duplicate (pure gather-friendly, FINN-legal
    `UpsampleNearestNeighbour`) + depthwise conv (FINN-legal `VVAU` via
    `InferVectorVectorActivation`) -- no cross-channel mixing anywhere,
    no overlap-add, unlike a depthwise ConvTranspose (blocked in FINN)."""

    def __init__(self, channels: int, bit_width: int):
        super().__init__()
        self.stem = _quant_conv2d(IN_CHANNELS, channels, bit_width, kernel_size=1)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.stem_act = _quant_act(bit_width)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.dw = _quant_conv2d(channels, channels, bit_width, kernel_size=3, padding=1, groups=channels)
        self.dw_bn = nn.BatchNorm2d(channels)
        self.dw_act = _quant_act(bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_act(self.stem_bn(self.stem(x)))
        x = self.upsample(x)
        x = self.dw_act(self.dw_bn(self.dw(x)))
        return x.value if hasattr(x, "value") else x


def main() -> None:
    torch.manual_seed(0)
    dummy = torch.rand(1, IN_CHANNELS, *INPUT_HW) * 2 - 1

    print("ProbeUpsampleDepthwise (Approach 2 bilinear-upsample decomposition) export")
    print(f"  channels={CHANNELS}, bit_width={BIT_WIDTH}, input=({IN_CHANNELS},{INPUT_HW[0]},{INPUT_HW[1]})")

    model = ProbeUpsampleDepthwise(CHANNELS, BIT_WIDTH).eval()

    with torch.no_grad():
        out = model(dummy)
    assert out.shape[2:] == (INPUT_HW[0] * 2, INPUT_HW[1] * 2), f"unexpected output HxW {tuple(out.shape[2:])}"
    assert torch.isfinite(out).all(), "output contains NaN/Inf -- weights not properly initialized"
    print(f"  forward OK: output shape {tuple(out.shape)}, finite (dummy weights confirmed live)")

    path = export_model(model, MODEL_NAME, dummy)
    print(f"\nSaved QONNX model to: {path}")


if __name__ == "__main__":
    main()
