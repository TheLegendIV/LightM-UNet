"""Export the UPSAMPLING-bottleneck probes (host / lightmunet_dev: torch + brevitas + qonnx).

Case = ENet up4-like block, nearest-neighbour decoder: Cin = 32 -> Cout = 16, internal ratio 4 (Cmid = Cin // 4 = 8), 32x32 input -> 64x64
output, uniform INT b, ReLU activations, dummy weights with torch.manual_seed(0). Model = QuantIdentity (unsigned; stand-in for the previous
block's Thr_out) -> FINNUpsamplingBottleneck. Two variants per bit width (6 probes):

  conv    decoder_type "nearest_conv_upsample": main = main_proj(1x1) -> nearest x2 -> skip_resize_conv (3x3)  [the decoder this repo trains]
  noconv  decoder_type "nearest_upsample":      main = main_proj(1x1) -> nearest x2                            [no 3x3 skip conv]

The ext branch (reduce 1x1 -> ConvTranspose 2x2 stride 2 -> expand 1x1) is identical. Frame budget used by the folding configs:
F = 18 * 64 * 64 = 73728 cycles (T_out = 18 per output pixel), the same F as the 72-cycle regular / downsampling probes.

    python3 hardware/builds/bottleneck_probe_v1/export_up_probe.py [--bits 4 6 8] [--variants conv noconv]
Writes inputs/up_cin32_cout16_in32_int{b}_{variant}.onnx and ..._probe.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "enet"))

import brevitas.nn as qnn  # noqa: E402
from brevitas.quant import Uint8ActPerTensorFloat  # noqa: E402
import nnunetv2.nets.LayerQuantEnetFINN as _fin  # noqa: E402
from nnunetv2.nets.QuantENet import _quant_act, _quant_block_act, _quant_conv2d  # noqa: E402

# the working-tree LayerQuantENet*.py currently lack their QuantENet import block (uncommitted edit); inject the helpers (see export_dn_probe.py)
for _n, _o in (("_quant_act", _quant_act), ("_quant_block_act", _quant_block_act), ("_quant_conv2d", _quant_conv2d)):
    if not hasattr(_fin, _n):
        setattr(_fin, _n, _o)

OUT_DIR = Path(__file__).resolve().parent / "inputs"
CIN, COUT, V, HW_IN, T_OUT = 32, 16, 4, 32, 18
BLOCKS = {"up4": (32, 16, 4, 32), "up5": (16, 4, 4, 64)}   # (Cin, Cout, internal ratio, input H=W) of ENet U4 = widths (4,16,32,16,4)
FRAME_CYCLES = 73728                                         # same frame budget as the 72-cycle 32x32 probes: T_out = F / (4*H*W)
DECODER = {"conv": "nearest_conv_upsample", "noconv": "nearest_upsample"}


class UpProbe(nn.Module):
    def __init__(self, bits: int, variant: str):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(bit_width=bits, act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)
        weight_bits = {"main_proj.0": bits, "skip_resize_conv.0": bits, "reduce.0": bits, "up.0": bits, "expand.0": bits}
        act_bits = {"residual_add": bits, "skip_resize_conv.2": bits, "reduce.2": bits, "up.2": bits, "out_act": bits}
        self.block = _fin.FINNUpsamplingBottleneck(CIN, COUT, weight_bits, act_bits, internal_ratio=V, decoder_type=DECODER[variant])
        for m in self.block.modules():
            if isinstance(m, nn.Dropout2d):
                m.p = 0.0

    def forward(self, x):
        return self.block(self.input_quant(x))


def case_name(bits: int, variant: str) -> str:
    return f"up_cin{CIN}_cout{COUT}_in{HW_IN}_int{bits}_{variant}"


def export_case(bits: int, variant: str) -> Path:
    from brevitas.export import export_qonnx
    from qonnx.core.datatype import DataType
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.util.cleanup import cleanup as qonnx_cleanup

    torch.manual_seed(0)
    model = UpProbe(bits, variant).cpu().eval()
    with torch.no_grad():
        model.train()
        for _ in range(3):
            model(torch.rand(4, CIN, HW_IN, HW_IN))
        model.eval()
    name = case_name(bits, variant)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.onnx"
    export_qonnx(model, export_path=str(path), input_t=torch.rand(1, CIN, HW_IN, HW_IN))
    qonnx_cleanup(str(path), out_file=str(path))
    qm = ModelWrapper(str(path))
    qm.set_tensor_datatype(qm.graph.input[0].name, DataType["INT8"])
    qm.set_tensor_datatype(qm.graph.output[0].name, DataType[f"UINT{bits}"])
    qm.save(str(path))
    ops: dict[str, int] = {}
    for n in qm.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    info = dict(
        name=name, block="up", cin=CIN, cout=COUT, v=V, cmid=CIN // V, height=HW_IN, width=HW_IN, hout=2 * HW_IN, wout=2 * HW_IN, bits=bits,
        seed=0, skip_conv=(variant == "conv"), decoder_type=DECODER[variant], T_out=T_OUT, F=int(round(T_OUT * (2 * HW_IN) ** 2)), op_counts=ops,
    )
    (OUT_DIR / f"{name}_probe.json").write_text(json.dumps(info, indent=2))
    print(f"{name}: {len(qm.graph.node)} nodes {dict(sorted(ops.items()))}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--variants", nargs="+", default=["conv", "noconv"], choices=["conv", "noconv"])
    ap.add_argument("--block", choices=sorted(BLOCKS), default="up4", help="up4: 32->16 at 32x32 in; up5: 16->4 at 64x64 in (U4 widths)")
    a = ap.parse_args()
    global CIN, COUT, V, HW_IN, T_OUT
    CIN, COUT, V, HW_IN = BLOCKS[a.block]
    T_OUT = FRAME_CYCLES / (4 * HW_IN ** 2)
    for b in a.bits:
        for v in a.variants:
            export_case(b, v)


if __name__ == "__main__":
    main()
