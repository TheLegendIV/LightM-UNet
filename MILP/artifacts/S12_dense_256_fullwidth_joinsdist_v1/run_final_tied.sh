#!/bin/bash
# Final full two-pass run (tied residual bits + LUTRAM joins are now always on): S12 dense 256x256 full width (config_12_dense_relu_nearest_conv_upsample_256), --joins-distributed,
# caps LUT 0.7 BRAM 0.4 DSP 0.9, bits 4/6/8, force-dsp, FPS 100 and DSR 1.04 (min feasible, see search_pass1/min_feasible_dsr.json)
# in BOTH passes. Pass 1 accuracy; pass 2 pins bits, min resources under pass 1's own use. Run in lightmunet_dev.
cd /workspace/LightM-UNet
OUT=MILP/artifacts/S12_dense_256_fullwidth_joinsdist_v1/final_tied_dsr1.04
python3 MILP/finn_milp.py --config config_12_dense_relu_nearest_conv_upsample_256 \
  --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_conv_upsample_256.json \
  --candidate-bits 4,6,8 --max-lut-fraction 0.7 --max-bram-fraction 0.4 --max-dsp-fraction 0.9 \
  --force-dsp --lexicographic --target-fps 100 --dsr-pct 4 \
  --time-limit 900 --gap-rel 0.005 --out-file $OUT/layer_bits_folding_final.json > $OUT/solve.log 2>&1
