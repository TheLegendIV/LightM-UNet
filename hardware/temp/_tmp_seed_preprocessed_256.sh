#!/bin/bash
set -e
SRC=/workspace/LightM-UNet/data/nnUNet_results/Dataset510_ARCADE_256_4c/nnUNetTrainerLayerQuantENet_12_dense_relu_nearest_conv_upsample_256_perlayer_12_dense_relu_nearest_conv_upsample_256_joint_alpha1.0_perlayer_candidatebits468_forcedsp_lut50_bram50_dsp90_fps250_ft15ep__nnUNetPlans__2d
DST=/workspace/LightM-UNet/data/nnUNet_preprocessed/Dataset510_ARCADE_256_4c
mkdir -p "$DST"
cp "$SRC/dataset.json" "$DST/dataset.json"
cp "$SRC/dataset_fingerprint.json" "$DST/dataset_fingerprint.json"
cp "$SRC/plans.json" "$DST/plans.json"
cp "$SRC/plans.json" "$DST/nnUNetPlans.json"
ls -la "$DST"
