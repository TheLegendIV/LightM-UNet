"""Fast (no-Vivado) preamble for the REAL TRAINED per-block HAWQ S19
checkpoint (quantEnet_s19_hawq_block_trained.onnx -- see
finn_export_s19_hawq_block_trained.py), identical in every way to
finn_hawq_preamble.py except MODEL_NAME -- same architecture/bit-widths as
the fresh-weight HAWQ export, so the conv_order.json sidecar
(quantEnet_s19_hawq_block_int8_conv_order.json) and folding_block_s19.json
bridge (finn_hawq_folding_bridge.py) both still apply unchanged.

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \
        /home/thelegendiv/finn/notebooks/enet/finn_hawq_preamble_trained.py
"""
import os
import sys
import dataclasses
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
import finn.builder.build_dataflow as build  # noqa: E402

MODEL_NAME = "quantEnet_s19_hawq_block_trained"
MODEL_FILE = os.path.join(base.ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(base.ENET_DIR, "finn_deployment_outputs", f"hawq_trained_preamble_{timestamp}")

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
    print("OUTPUT_DIR=", OUTPUT_DIR, flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    build.build_dataflow_cfg(MODEL_FILE, cfg)
    print("Done. Intermediate models in:", os.path.join(OUTPUT_DIR, "intermediate_models"))
    print("OUTPUT_DIR=", OUTPUT_DIR)
