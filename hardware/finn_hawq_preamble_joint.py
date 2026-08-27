"""Same as finn_hawq_preamble.py, but for the JOINT (w,a)-pair HAWQ bit-width
export (quantEnet_s19_hawq_block_joint_int8.onnx, built from
compression/hawq/block_bits_s19_joint.json via
`finn_export_s19_hawq_block.py --block-bits-file .../block_bits_s19_joint.json`)
instead of the per-axis two-pass "block" scheme. Architecture is identical
(same BLOCK_NAMES/module structure) -- only the weight_bits/act_bits per
block differ -- so this preamble is otherwise a byte-for-byte copy of
finn_hawq_preamble.py with just MODEL_NAME/OUTPUT_DIR changed.

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \
        /home/thelegendiv/finn/notebooks/enet/finn_hawq_preamble_joint.py
"""
import os
import sys
import dataclasses
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
import finn.builder.build_dataflow as build  # noqa: E402

MODEL_NAME = "quantEnet_s19_hawq_block_joint_int8"
MODEL_FILE = os.path.join(base.ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(base.ENET_DIR, "finn_deployment_outputs", f"hawq_joint_preamble_{timestamp}")

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
