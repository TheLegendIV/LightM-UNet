"""Preamble (no-Vivado) for the REAL, QAT fine-tuned checkpoint_best.pth
(epoch 50) weight transfer of the per-block-HAWQ-bit-width (min4 scheme),
separable_dilated+dense_dilation, plain-ReLU 12_separable_dense_relu export
(quantEnet_12_separable_dense_relu_min4_hawq_trained_int8.onnx, built via
finn_export_12_separable_dense_relu_min4_hawq_trained.py from
compression/hawq/artifacts/block_bits_12_separable_dense_relu_min4.json +
data/nnUNet_results/.../nnUNetTrainerCombinedQuantENet_12_separable_dense_relu_
perblock_12_separable_dense_relu_min4_ft50ep__nnUNetPlans__2d/fold_0/
checkpoint_best.pth).

Byte-for-byte copy of finn_hawq_preamble_12_separable_dense_relu_min4.py
(the DUMMY-weight sibling) with only MODEL_NAME/OUTPUT_DIR changed -- the
tidy/streamline/convert_to_hw/8-way-partition-assignment steps are purely
structural, but MUST be re-run against this ONNX (not reused from the
dummy preamble's own outputs): real trained weights produce different
Quant node scale constants, which changes streamlining's constant-folding
results even though the architecture graph shape is identical.

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_hawq_preamble_12_separable_dense_relu_min4_trained.py
"""
import os
import sys
import dataclasses
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
import finn.builder.build_dataflow as build  # noqa: E402

MODEL_NAME = "quantEnet_12_separable_dense_relu_min4_hawq_trained_int8"
MODEL_FILE = os.path.join(base.ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(base.ENET_DIR, "finn_deployment_outputs", f"hawq_12_sep_dense_relu_min4_trained_preamble_{timestamp}")

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
