"""Export the DOWNSAMPLING-bottleneck probes (host / lightmunet_dev: torch + brevitas + qonnx).

Case = ENet down2-like block: Cin = 16 -> Cout = 32, internal_ratio 4 (Cmid = 8), 64x64 input -> 32x32 output, uniform INT b,
ReLU activations, dummy weights with torch.manual_seed(0). Model = QuantIdentity (unsigned; stand-in for the previous block's
Thr_out) -> downsampling bottleneck. Two skip variants per bit width (6 probes total):

  mvau   FINNDownsamplingBottleneck unchanged: skip = MaxPool -> frozen padded-identity 1x1 conv (INT8 weights) -> residual add.
  fmpad  same block, but the identity conv is replaced by a channel zero-pad on the maxpool output
         (F.pad on the channel axis; the residual add's shared input quantizer then requantizes the padded skip, i.e. the
         "pad_thr" order of dn_bottleneck.py). FINN needs a custom pass to turn that Pad into an FMPadding on a regrouped stream
         (hardware/finn_channel_pad.py, to be written in the FINN container, see FINN_AGENT_HANDOFF.md).

    python3 hardware/builds/bottleneck_probe_v1/export_dn_probe.py [--bits 4 6 8] [--variants fmpad mvau]
Writes inputs/dn_cin16_cout32_in64_int{b}_{variant}.onnx and ..._probe.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "enet"))

import brevitas.nn as qnn  # noqa: E402
from brevitas.quant import Uint8ActPerTensorFloat  # noqa: E402
import nnunetv2.nets.LayerQuantEnetFINN as _fin  # noqa: E402
from nnunetv2.nets.QuantENet import _quant_act, _quant_block_act, _quant_conv2d  # noqa: E402

# The working-tree LayerQuantENet*.py currently lack their `from nnunetv2.nets.QuantENet import (...)` blocks (an uncommitted edit
# deleted them; HEAD has them). Inject the helpers so the probe works either way.
for _mod in (_fin,):
    for _n, _o in (("_quant_act", _quant_act), ("_quant_block_act", _quant_block_act), ("_quant_conv2d", _quant_conv2d)):
        if not hasattr(_mod, _n):
            setattr(_mod, _n, _o)

OUT_DIR = Path(__file__).resolve().parent / "inputs"
CIN, COUT, V, HW_IN, T_OUT = 16, 32, 4, 64, 72
SKIP_ORDER = {"fmpad": "pad_thr", "mvau": "pad_thr"}   # export order of the skip requantizer (before FINN streamlining)


class PadDownsamplingBottleneck(_fin.FINNDownsamplingBottleneck):
    """FINNDownsamplingBottleneck with the padded-identity 1x1 conv replaced by a plain channel zero-pad."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.shortcut_proj                      # no MVAU on the skip
        self._pad_c = self.expand[0].out_channels - self.reduce[0].in_channels

    def forward(self, x):
        pooled = self.shortcut_pool(x)
        pooled = getattr(pooled, "value", pooled)  # QuantTensor -> tensor; the add's shared input quantizer requantizes it
        skip = F.pad(pooled, (0, 0, 0, 0, 0, self._pad_c))   # NCHW channel axis, constant 0
        out = self.dropout(self.expand(self.conv(self.reduce(x))))
        return self.out_act(self.residual_add(skip, out))


class DnProbe(nn.Module):
    def __init__(self, bits: int, variant: str):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(bit_width=bits, act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)
        weight_bits = {"reduce.0": bits, "conv.0": bits, "expand.0": bits}
        act_bits = {"reduce.2": bits, "conv.2": bits, "residual_add": bits, "out_act": bits}
        cls = PadDownsamplingBottleneck if variant == "fmpad" else _fin.FINNDownsamplingBottleneck
        self.block = cls(CIN, COUT, weight_bits, act_bits, internal_ratio=V, dropout_p=0.0)

    def forward(self, x):
        return self.block(self.input_quant(x))


def case_name(bits: int, variant: str) -> str:
    return f"dn_cin{CIN}_cout{COUT}_in{HW_IN}_int{bits}_{variant}"


def export_case(bits: int, variant: str) -> Path:
    from brevitas.export import export_qonnx
    from qonnx.core.datatype import DataType
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.util.cleanup import cleanup as qonnx_cleanup

    torch.manual_seed(0)
    model = DnProbe(bits, variant).cpu().eval()
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
        name=name, block="down", cin=CIN, cout=COUT, v=V, cmid=COUT // V, height=HW_IN, width=HW_IN, hout=HW_IN // 2, wout=HW_IN // 2,
        bits=bits, seed=0, skip_pad=variant, skip_order=SKIP_ORDER[variant], T_out=T_OUT, F=T_OUT * (HW_IN // 2) ** 2, op_counts=ops,
    )
    (OUT_DIR / f"{name}_probe.json").write_text(json.dumps(info, indent=2))
    print(f"{name}: {len(qm.graph.node)} nodes {dict(sorted(ops.items()))}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--variants", nargs="+", default=["fmpad", "mvau"], choices=["fmpad", "mvau"])
    a = ap.parse_args()
    for b in a.bits:
        for v in a.variants:
            export_case(b, v)


if __name__ == "__main__":
    main()
