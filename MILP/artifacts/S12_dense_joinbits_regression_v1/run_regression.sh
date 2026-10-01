#!/bin/bash
# Regression of the T=1024 thresholding cost model + --tie-residual-bits on S12 512x512 (config_12_dense_relu_nearest_upsample_wm).
# FPS 250, hard caps 0.7 (LUT/BRAM/DSP), --mvau-wwidth-max UNSET, lexicographic: pass 1 accuracy (bits 4,6,8, no DSR),
# pass 2 min resources at DSR = 1 + epsilon. fixed = legacy Int8 joins; tied = skip_quant/residual_add bits = expand.0 act bits.
# Run in lightmunet_dev from /workspace/LightM-UNet.  Usage: run_regression.sh [dsr2 ...]   (default: 1.02)
cd /workspace/LightM-UNet
OUT=MILP/artifacts/S12_dense_joinbits_regression_v1
DSRS=${@:-1.02}
COMMON="--config config_12_dense_relu_nearest_upsample_wm \
  --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_upsample_wm.json \
  --candidate-bits 4,6,8 --hard-lut-fraction 0.7 --hard-bram-fraction 0.7 --hard-dsp-fraction 0.7 \
  --force-dsp --target-fps 250 --lexicographic --time-limit 900 --gap-rel 0.005"
for D in $DSRS; do
  for ARM in fixed tied; do
    TAG=${ARM}_dsr2_${D}
    EXTRA=""; [ $ARM = tied ] && EXTRA="--tie-residual-bits"
    mkdir -p $OUT/$TAG
    python3 MILP/finn_milp.py $COMMON $EXTRA --dsr-ratio-pass2 $D \
      --out-file $OUT/$TAG/layer_bits_folding_$TAG.json > $OUT/$TAG/solve.log 2>&1 &
  done
done
wait
