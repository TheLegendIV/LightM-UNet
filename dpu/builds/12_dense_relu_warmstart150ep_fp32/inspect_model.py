"""Vitis AI Model Inspector: which ops map to the DPU target, which fall back to CPU.

Named inspect_model.py (not inspect.py) so it does not shadow the stdlib module.
Runs inside the Vitis AI container; see run_flow.sh.
"""
import os
import sys

import torch
from pytorch_nndct.apis import Inspector

sys.path.insert(0, "/job")
from model import build_model, input_shape  # noqa: E402

target = os.environ["INSPECT_TARGET"]
out_dir = "/workspace/inspect"

model = build_model().eval()
dummy = torch.randn(*input_shape())

inspector = Inspector(target)
inspector.inspect(model, (dummy,), device=torch.device("cpu"),
                  output_dir=out_dir, image_format="svg")
print(f"Inspector report written to {out_dir}")
