"""Preamble variant of finn_hawq_preamble_12_dense_relu_warmstart150ep_
alpha025_trained.py pointed at the FINE-TUNED main_up export
(quantEnet_12_dense_relu_warmstart150ep_alpha025_trained_ftmainup_int8.onnx,
built via finetune_main_up_12_dense_relu_warmstart150ep_alpha025.py) --
used to check whether perturbing up4/up5.main_up's weight away from the
exact, mostly-zero block-diagonal bilinear kernel (via a short feature-
matching fine-tune, everything else frozen) avoids the dangling-tensor
FINN-internals bug ("Bug #3", see
memories/repo/finn_12_dense_relu_alpha025_perlayer.md) that the exact
frozen-kernel version hit in step_enet_convert_to_hw.

Byte-for-byte copy of the _trained.py preamble with only MODEL_NAME/
OUTPUT_DIR changed.

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_hawq_preamble_12_dense_relu_warmstart150ep_alpha025_trained_ftmainup.py
"""
import os
import sys
import dataclasses
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
import finn.builder.build_dataflow as build  # noqa: E402

MODEL_NAME = "quantEnet_12_dense_relu_warmstart150ep_alpha025_trained_ftmainup_int8"
MODEL_FILE = os.path.join(base.ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(base.ENET_DIR, "finn_deployment_outputs", f"12_dense_relu_warmstart150ep_alpha025_trained_ftmainup_preamble_{timestamp}")

idx = base.enet_ip_partitioned_8way_steps.index(base.assign_stage_partition_ids_8way)
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
