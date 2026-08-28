"""Fast (no-Vivado) preamble for the fresh-weight, per-block-HAWQ-bit-width,
per-block-leaky-ReLU 26_5_w24 JOINT-scheme export
(quantEnet_26_5_w24_hawq_joint_int8.onnx, built via
finn_export_26_5_w24_hawq_joint.py from
compression/hawq/block_bits_26_5_w24_acc1x_joint.json).

Byte-for-byte copy of finn_hawq_preamble_joint.py (itself a copy of
finn_hawq_preamble.py) with only MODEL_NAME/OUTPUT_DIR changed -- the
tidy/streamline/convert_to_hw/8-way-partition-assignment steps are all
purely structural (see finn_stage_partition.py's docstring: keyed off
StreamingMaxPool/FMPadding_Pixel node COUNTS, not channel widths or
per-stage depths), so they apply unchanged to this differently-shaped
(CHANNELS=(4,8,24,8,4), BOTTLENECKS_PER_STAGE=(4,8,8,2,1)) architecture.

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \
        /home/thelegendiv/finn/notebooks/enet/finn_hawq_preamble_26_5_w24_joint.py
"""
import os
import sys
import dataclasses
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
import finn.builder.build_dataflow as build  # noqa: E402

MODEL_NAME = "quantEnet_26_5_w24_hawq_joint_int8"
MODEL_FILE = os.path.join(base.ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(base.ENET_DIR, "finn_deployment_outputs", f"hawq_26_5_w24_joint_preamble_{timestamp}")

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
