"""Export the FINAL-deconvolution probes (host / lightmunet_dev: torch + brevitas + qonnx).

Case = ENet U4 final layer of LayerQuantEnetFINN: qnn.QuantConvTranspose2d(c5 = 4 -> out_channels = 5, kernel 2, stride 2, Int8 weight quantizer
at weight_bit_width = b), 128x128 input -> 256x256 output, dummy weights with torch.manual_seed(0). Model = QuantIdentity (unsigned; stand-in for
regular5's Thr_out) -> final layer. Two variants per bit width (6 probes):

  bias    bias=True, bias_quant=Int32Bias  [LayerQuantEnetFINN default, final_bias=True]
  nobias  bias=False                        [what hardware/finn_enet_prod_export.py exports today: "bias requires extra handling"]

The output is the raw logit (no output quantizer, no threshold); the graph output is tagged INT8 like finn_enet_prod_export.export_model does.
Frame budget used by the folding configs: F = 73728 cycles (T_out = 1.125 per output pixel), the same F as the other U4 probes.

    python3 hardware/builds/bottleneck_probe_v1/export_fnl_probe.py [--bits 4 6 8] [--variants bias nobias]
Writes inputs/fnl_cin4_cout5_in128_int{b}_{variant}.onnx and ..._probe.json.
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
from brevitas.quant import Int8WeightPerTensorFloat, Int32Bias, Uint8ActPerTensorFloat  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "inputs"
CIN, COUT, HW_IN, F = 4, 5, 128, 73728


class FnlProbe(nn.Module):
    def __init__(self, bits: int, bias: bool):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(bit_width=bits, act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)
        self.final = qnn.QuantConvTranspose2d(
            CIN, COUT, kernel_size=2, stride=2, bias=bias, bias_quant=Int32Bias if bias else None,
            weight_bit_width=bits, weight_quant=Int8WeightPerTensorFloat,
        )

    def forward(self, x):
        out = self.final(self.input_quant(x))
        return out.value if hasattr(out, "value") else out


def case_name(bits: int, variant: str) -> str:
    return f"fnl_cin{CIN}_cout{COUT}_in{HW_IN}_int{bits}_{variant}"


def export_case(bits: int, variant: str) -> Path:
    from brevitas.export import export_qonnx
    from qonnx.core.datatype import DataType
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.util.cleanup import cleanup as qonnx_cleanup

    torch.manual_seed(0)
    model = FnlProbe(bits, variant == "bias").cpu().eval()
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
    qm.set_tensor_datatype(qm.graph.output[0].name, DataType["INT8"])
    qm.save(str(path))
    ops: dict[str, int] = {}
    for n in qm.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    info = dict(name=name, block="final", cin=CIN, cout=COUT, cmid=CIN, height=HW_IN, width=HW_IN, hout=2 * HW_IN, wout=2 * HW_IN, bits=bits, seed=0,
                bias=(variant == "bias"), T_out=F / (2 * HW_IN) ** 2, F=F, op_counts=ops)
    (OUT_DIR / f"{name}_probe.json").write_text(json.dumps(info, indent=2))
    print(f"{name}: {len(qm.graph.node)} nodes {dict(sorted(ops.items()))}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--variants", nargs="+", default=["bias", "nobias"], choices=["bias", "nobias"])
    a = ap.parse_args()
    for b in a.bits:
        for v in a.variants:
            export_case(b, v)


if __name__ == "__main__":
    main()
