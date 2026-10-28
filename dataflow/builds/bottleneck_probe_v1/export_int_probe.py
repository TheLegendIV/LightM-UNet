"""Export the INITIAL-block probes (host / lightmunet_dev: torch + brevitas + qonnx).

Case = ENet U4 initial block, FINNInitialBlockConcat from LayerQuantEnetFINN.py: 1 -> 4 channels (conv branch 3 channels, 3x3 stride 2 pad 1;
maxpool branch 1 channel, 2x2 stride 2; shared branch_quant; concat; BN + ReLU), 256x256 input -> 128x128 output, uniform INT b, dummy weights
(torch.manual_seed(0)). The block has its own input quantizer, so no stand-in is needed. Frame budget of the folding config: F = 81920 cycles
(T_out = 5.0 per output pixel) = the maxpool floor 1.25 * 256 * 256 of this input size.

    python3 hardware/builds/bottleneck_probe_v1/export_int_probe.py [--bits 4 6 8]
Writes inputs/init_cin1_cout4_in256_int{b}.onnx and ..._probe.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "enet"))

import nnunetv2.nets.LayerQuantEnetFINN as _fin  # noqa: E402
from nnunetv2.nets.QuantENet import _quant_act, _quant_block_act, _quant_conv2d  # noqa: E402

for _n, _o in (("_quant_act", _quant_act), ("_quant_block_act", _quant_block_act), ("_quant_conv2d", _quant_conv2d)):
    if not hasattr(_fin, _n):
        setattr(_fin, _n, _o)

OUT_DIR = Path(__file__).resolve().parent / "inputs"
CIN, COUT, HW_IN, F = 1, 4, 256, 81920


def case_name(bits: int) -> str:
    return f"init_cin{CIN}_cout{COUT}_in{HW_IN}_int{bits}"


def export_case(bits: int) -> Path:
    from brevitas.export import export_qonnx
    from qonnx.core.datatype import DataType
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.util.cleanup import cleanup as qonnx_cleanup

    torch.manual_seed(0)
    model = _fin.FINNInitialBlockConcat(CIN, COUT, {"conv": bits}, {"input_quant": bits, "act": bits}).cpu().eval()
    with torch.no_grad():
        model.train()
        for _ in range(3):
            model(torch.rand(4, CIN, HW_IN, HW_IN))
        model.eval()
    name = case_name(bits)
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
    info = dict(name=name, block="init", cin=CIN, cout=COUT, cmid=COUT - CIN, height=HW_IN, width=HW_IN, hout=HW_IN // 2, wout=HW_IN // 2, bits=bits,
                seed=0, T_out=F / (HW_IN // 2) ** 2, F=F, op_counts=ops)
    (OUT_DIR / f"{name}_probe.json").write_text(json.dumps(info, indent=2))
    print(f"{name}: {len(qm.graph.node)} nodes {dict(sorted(ops.items()))}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8])
    a = ap.parse_args()
    for b in a.bits:
        export_case(b)


if __name__ == "__main__":
    main()
