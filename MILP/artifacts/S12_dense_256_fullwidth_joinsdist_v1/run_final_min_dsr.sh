#!/bin/bash
# Refactor check + deliverable form: S12 dense 256x256 (config_12_dense_relu_nearest_conv_upsample_256, channels 4,16,32,16,4),
# caps LUT 0.7 BRAM 0.4 DSP 0.9, bits 4/6/8, force-dsp, FPS 100, --min-dsr (replaces the manual DSR bisect), lexicographic.
# Expected to land on the earlier tied run (DSR 1.04). Run in lightmunet_dev from /workspace/LightM-UNet.
cd /workspace/LightM-UNet
OUT=MILP/artifacts/S12_dense_256_fullwidth_joinsdist_v1/final_min_dsr
python3 MILP/finn_milp.py --config config_12_dense_relu_nearest_conv_upsample_256 \
  --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_conv_upsample_256.json \
  --candidate-bits 4,6,8 --max-lut-fraction 0.7 --max-bram-fraction 0.4 --max-dsp-fraction 0.9 \
  --force-dsp --lexicographic --target-fps 100 --min-dsr \
  --time-limit 900 --gap-rel 0.005 --out-file $OUT/layer_bits_folding_final.json > $OUT/solve.log 2>&1
