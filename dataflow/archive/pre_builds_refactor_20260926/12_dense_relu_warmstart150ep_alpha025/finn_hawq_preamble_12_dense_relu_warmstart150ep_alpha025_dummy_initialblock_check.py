"""Throwaway validation preamble: runs quantEnet_12_dense_relu_warmstart150ep
_alpha025_dummy_int8.onnx (fresh weights, but includes the new
FINNInitialBlockConcat substitute -- see finn_export_12_dense_relu_
warmstart150ep_alpha025_dummy.py's module docstring point 0) through tidy/
streamline/convert_to_hw only, to check whether Concat_0 (and the
LayerQuantInitialBlock region's BatchNorm/MaxPool/Transpose nodes) now
convert to HW nodes, before spending any time on the real trained export.

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_hawq_preamble_12_dense_relu_warmstart150ep_alpha025_dummy_initialblock_check.py
"""
import os
import sys
import dataclasses
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
import finn.builder.build_dataflow as build  # noqa: E402

MODEL_NAME = "quantEnet_12_dense_relu_warmstart150ep_alpha025_dummy_int8"
MODEL_FILE = os.path.join(base.ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(base.ENET_DIR, "finn_deployment_outputs", f"12_dense_relu_warmstart150ep_alpha025_dummy_initialblock_check_{timestamp}")

idx = base.enet_ip_partitioned_8way_steps.index(base.step_enet_convert_to_hw)
steps = base.enet_ip_partitioned_8way_steps[: idx + 1]
print("Steps to run:", [s if isinstance(s, str) else s.__name__ for s in steps])

cfg = dataclasses.replace(
    base.cfg_stitched_ip_partitioned_8way,
    output_dir=OUTPUT_DIR,
    steps=steps,
    generate_outputs=[],
    save_intermediate_models=True,
)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("OUTPUT_DIR=", OUTPUT_DIR, flush=True)
    build.build_dataflow_cfg(MODEL_FILE, cfg)
    print("Done. Intermediate models in:", os.path.join(OUTPUT_DIR, "intermediate_models"))
    print("OUTPUT_DIR=", OUTPUT_DIR)
