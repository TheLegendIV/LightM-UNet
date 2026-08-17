"""Diagnostic: does brevitas export_qonnx emit a Dropout node for eval-mode
nn.Dropout2d, and (if so) does qonnx_cleanup() remove it? Checks the RAW
export (pre-cleanup) node list directly."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_deps"))
from finn_enet_prod_export import FINNQuantENet  # noqa: E402

from brevitas.export import export_qonnx
import onnx

dummy = torch.rand(1, 1, 64, 64) * 2 - 1
model = FINNQuantENet(in_channels=1, out_channels=2, channels=(20, 72, 144, 72, 20),
                       bottlenecks_per_stage=(1, 1, 1, 1, 1), use_dilated=True,
                       bit_width=8, residual=True).eval()

raw_path = "/tmp/_dropout_check_raw.onnx"
export_qonnx(model, export_path=raw_path, input_t=dummy)

raw = onnx.load(raw_path)
ops = {}
for n in raw.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("RAW export (pre-cleanup) node count:", len(raw.graph.node))
print("RAW op histogram:", dict(sorted(ops.items())))
print("Dropout present in RAW export:", any("dropout" in k.lower() for k in ops))

from qonnx.util.cleanup import cleanup as qonnx_cleanup
clean_path = "/tmp/_dropout_check_clean.onnx"
qonnx_cleanup(raw_path, out_file=clean_path)
clean = onnx.load(clean_path)
ops2 = {}
for n in clean.graph.node:
    ops2[n.op_type] = ops2.get(n.op_type, 0) + 1
print("\nCLEANED node count:", len(clean.graph.node))
print("CLEANED op histogram:", dict(sorted(ops2.items())))
print("Dropout present after cleanup:", any("dropout" in k.lower() for k in ops2))
