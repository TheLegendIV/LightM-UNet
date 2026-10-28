"""Export one QONNX model per bottleneck probe case (host / lightmunet_dev: needs torch + brevitas + qonnx).

Case = one real ENet regular bottleneck (LayerQuantRegularBottleneck, the class LayerQuantEnetFINN reuses unmodified):
Cin = Cout = 32, internal_ratio 4 (Cmid = 8), dense 3x3, dilation d, padding d (same), 32x32, uniform INT b,
ReLU activations. Dummy weights, torch.manual_seed(0) for every case (reproducible, see README "Weight noise").

Model = QuantIdentity (unsigned, b bits; stand-in for the previous block's Thr_out) -> bottleneck.
So the FINN graph is:  Thr_in -> Dup -> reduce/conv/expand MVAUs (+ thresholds) -> Add -> Thr(residual_add) -> Thr(out_act).

Run (from repo root, inside lightmunet_dev):
    python3 hardware/builds/bottleneck_probe_v1/export_bottleneck_probe.py [--dilations 1 2 4 8 16] [--bits 4 6 8]
Writes inputs/bottleneck_cin32_d{d}_int{b}.onnx and inputs/bottleneck_cin32_d{d}_int{b}_probe.json.
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
import nnunetv2.nets.LayerQuantENet as _lq  # noqa: E402
from nnunetv2.nets.QuantENet import _quant_block_act, _quant_conv2d  # noqa: E402

# The working-tree LayerQuantENet.py currently lacks its `from nnunetv2.nets.QuantENet import (...)` block (an uncommitted
# edit deleted it; HEAD has it). Inject the two helpers the regular bottleneck needs so this probe works either way.
for _name, _obj in (("_quant_block_act", _quant_block_act), ("_quant_conv2d", _quant_conv2d)):
    if not hasattr(_lq, _name):
        setattr(_lq, _name, _obj)
LayerQuantRegularBottleneck = _lq.LayerQuantRegularBottleneck

OUT_DIR = Path(__file__).resolve().parent / "inputs"
CIN = 32
INTERNAL_RATIO = 4
KERNEL = 3
HW = 32


class ProbeBlock(nn.Module):
    def __init__(self, bits: int, dilation: int):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(bit_width=bits, act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)
        weight_bits = {"reduce.0": bits, "conv": bits, "expand.0": bits}
        act_bits = {"reduce.2": bits, "conv_bn_act.2": bits, "residual_add": bits, "out_act": bits}
        self.block = LayerQuantRegularBottleneck(
            channels=CIN, weight_bits=weight_bits, act_bits=act_bits, internal_ratio=INTERNAL_RATIO,
            kernel_size=KERNEL, padding=dilation, dilation=dilation, asymmetric=False, use_dsc=False,
            separable_dilated=False, dropout_p=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.input_quant(x))


def case_name(dilation: int, bits: int) -> str:
    return f"bottleneck_cin{CIN}_d{dilation}_int{bits}"


def export_case(dilation: int, bits: int) -> Path:
    from brevitas.export import export_qonnx
    from qonnx.core.datatype import DataType
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.util.cleanup import cleanup as qonnx_cleanup

    torch.manual_seed(0)
    model = ProbeBlock(bits, dilation).cpu().eval()
    # the BN running stats / quantizer scales of a never-trained model are defaults; one calibration-style
    # forward in train mode fixes the activation scales so thresholds are non-degenerate
    with torch.no_grad():
        model.train()
        for _ in range(3):
            model(torch.rand(4, CIN, HW, HW))
        model.eval()

    name = case_name(dilation, bits)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.onnx"
    export_qonnx(model, export_path=str(path), input_t=torch.rand(1, CIN, HW, HW))
    qonnx_cleanup(str(path), out_file=str(path))
    qm = ModelWrapper(str(path))
    qm.set_tensor_datatype(qm.graph.input[0].name, DataType["INT8"])
    qm.set_tensor_datatype(qm.graph.output[0].name, DataType[f"UINT{bits}"])  # out_act is an unsigned ReLU quantizer
    qm.save(str(path))

    ops: dict[str, int] = {}
    for n in qm.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    info = dict(
        name=name, cin=CIN, v=INTERNAL_RATIO, z=INTERNAL_RATIO, cmid=CIN // INTERNAL_RATIO, cout=CIN, k=KERNEL,
        dilation=dilation, padding=dilation, stride=1, height=HW, width=HW, bits=bits, seed=0, op_counts=ops,
    )
    (OUT_DIR / f"{name}_probe.json").write_text(json.dumps(info, indent=2))
    print(f"{name}: {len(qm.graph.node)} nodes {dict(sorted(ops.items()))}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8])
    a = ap.parse_args()
    for d in a.dilations:
        for b in a.bits:
            export_case(d, b)


if __name__ == "__main__":
    main()
