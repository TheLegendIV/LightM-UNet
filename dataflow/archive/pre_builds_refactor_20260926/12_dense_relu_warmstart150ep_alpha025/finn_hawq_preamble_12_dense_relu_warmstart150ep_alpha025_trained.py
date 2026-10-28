"""Preamble (no-Vivado) for the REAL, QAT fine-tuned checkpoint (ft15ep)
weight transfer of the PER-LAYER-HAWQ-bit-width (alpha=0.25), dense
KxK-dilated-context (SEPARABLE_DILATED=False), plain-ReLU
12_dense_relu_warmstart150ep export
(quantEnet_12_dense_relu_warmstart150ep_alpha025_trained_int8.onnx, built
via finn_export_12_dense_relu_warmstart150ep_alpha025_trained.py from
compression/hawq/artifacts/12_dense_relu_warmstart150ep_ILP_outputs_
perlayer_forcedsp_lut70/layer_bits_SITES_..._alpha0.25_....json +
data/nnUNet_results/.../nnUNetTrainerLayerQuantENet_12_dense_relu_
warmstart150ep_perlayer_..._ft15ep__nnUNetPlans__2d/fold_0/
checkpoint_best.pth).

Byte-for-byte copy of finn_hawq_preamble_12_separable_dense_relu_min4_trained.py
(the per-BLOCK-HAWQ sibling family) with only MODEL_NAME/OUTPUT_DIR changed
-- the tidy/streamline/convert_to_hw/8-way-partition-assignment steps are
purely structural (identical stage naming: initial/down1/regular1/down2/
stage2/stage3/up4/regular4/up5/regular5/final), so they apply unchanged
here despite the different per-layer bit-width granularity.

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_hawq_preamble_12_dense_relu_warmstart150ep_alpha025_trained.py
"""
import os
import sys
import dataclasses
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
import finn.builder.build_dataflow as build  # noqa: E402

MODEL_NAME = "quantEnet_12_dense_relu_warmstart150ep_alpha025_trained_int8"
MODEL_FILE = os.path.join(base.ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(base.ENET_DIR, "finn_deployment_outputs", f"12_dense_relu_warmstart150ep_alpha025_trained_preamble_{timestamp}")

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
