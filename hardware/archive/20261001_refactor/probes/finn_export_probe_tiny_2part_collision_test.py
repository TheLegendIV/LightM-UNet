"""Export a MINIMAL self-contained residual probe (DuplicateStreams fork +
AddStreams join + 3 MVAU convs) used to validate the multi-partition VLNV
rename+repackage combine fix documented in finn_gotchas.md (2026-09-20 entry)
BEFORE committing to the full 512x512 8-way rebuild.

This SAME model is built TWICE independently (see
finn_build_probe_tiny_2part_collision_test.py, run once per OUTPUT_DIR
suffix "p0"/"p1") -- each independent FINN build/CreateStitchedIP run
restarts its own child-HLS-IP numbering from 0 (DuplicateStreams_hls_0,
AddStreams_hls_0, MVAU_hls_0/1/2), exactly reproducing the real
multi-partition VLNV collision class on a tiny, fast-to-synthesize model.

Fully self-contained -- only needs brevitas+qonnx (already present inside
the FINN container), no dependency on enet/nnunetv2.

Usage (inside the FINN container):
    python3 finn_export_probe_tiny_2part_collision_test.py
"""
from pathlib import Path

import torch
from torch import nn
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat

OUT_DIR = Path(__file__).resolve().parent

CHANNELS = 4
BIT_WIDTH = 8
IN_CHANNELS = 1
INPUT_HW = (8, 8)
MODEL_NAME = "quant_probe_tiny_2part_collision_test"


def _quant_conv2d(in_ch: int, out_ch: int, bit_width: int, **kwargs) -> qnn.QuantConv2d:
    return qnn.QuantConv2d(
        in_ch, out_ch, bias=False,
        weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat,
        **kwargs,
    )


def _quant_act(bit_width: int) -> qnn.QuantReLU:
    return qnn.QuantReLU(act_quant=Uint8ActPerTensorFloat, bit_width=bit_width)


def export_model(model: nn.Module, name: str, dummy: torch.Tensor) -> Path:
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


class TinyResidualProbe(nn.Module):
    """stem(1x1 conv) -> [fork: identity + 3x3 conv branch] -> add -> tail(1x1 conv).
    Guarantees DuplicateStreams + AddStreams + 3x MVAU (stem/branch/tail),
    the same op-type mix responsible for real cross-partition VLNV
    collisions in the S12-dense 8-way builds."""

    def __init__(self, channels: int, bit_width: int):
        super().__init__()
        self.stem = _quant_conv2d(IN_CHANNELS, channels, bit_width, kernel_size=1)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.stem_act = _quant_act(bit_width)

        self.branch = _quant_conv2d(channels, channels, bit_width, kernel_size=3, padding=1)
        self.branch_bn = nn.BatchNorm2d(channels)
        self.branch_act = _quant_act(bit_width)

        self.tail = _quant_conv2d(channels, channels, bit_width, kernel_size=1)
        self.tail_bn = nn.BatchNorm2d(channels)
        self.tail_act = _quant_act(bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_act(self.stem_bn(self.stem(x)))
        residual = x
        branch = self.branch_act(self.branch_bn(self.branch(x)))
        x = residual + branch
        x = self.tail_act(self.tail_bn(self.tail(x)))
        return x.value if hasattr(x, "value") else x


def main() -> None:
    torch.manual_seed(0)
    dummy = torch.rand(1, IN_CHANNELS, *INPUT_HW) * 2 - 1

    print("TinyResidualProbe (DuplicateStreams+AddStreams collision-test probe) export")
    print(f"  channels={CHANNELS}, bit_width={BIT_WIDTH}, input=({IN_CHANNELS},{INPUT_HW[0]},{INPUT_HW[1]})")

    model = TinyResidualProbe(CHANNELS, BIT_WIDTH).eval()

    with torch.no_grad():
        out = model(dummy)
    assert out.shape[2:] == INPUT_HW, f"unexpected output HxW {tuple(out.shape[2:])}"
    assert torch.isfinite(out).all(), "output contains NaN/Inf -- weights not properly initialized"
    print(f"  forward OK: output shape {tuple(out.shape)}, finite (dummy weights confirmed live)")

    path = export_model(model, MODEL_NAME, dummy)
    print(f"\nSaved QONNX model to: {path}")


if __name__ == "__main__":
    main()
