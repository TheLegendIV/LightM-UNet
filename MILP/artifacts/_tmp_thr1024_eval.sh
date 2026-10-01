#!/bin/bash
# One-build check of the T=1024 thresholding cost model: dsrmin_2x config (uniform INT4, DSR 2.04, FPS 305.17,
# min-resources), joins at 8-bit (current) vs 4-bit (what-if). Run in lightmunet_dev.
cd /workspace/LightM-UNet
OUT=MILP/artifacts/_tmp_thr1024_eval
COMMON="--config config_12_dense_relu_nearest_upsample_wm \
  --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_upsample_wm.json \
  --candidate-bits 4 --hard-lut-fraction 1.0 --hard-bram-fraction 1.0 --hard-dsp-fraction 1.0 \
  --hard-uram-fraction 1.0 --force-dsp --target-fps 305.17 --min-resources --time-limit 600 --gap-rel 0.001 --mvau-wwidth-max 80 --dsr-ratio 2.04"
for JB in 8 4; do
  mkdir -p $OUT/joins$JB
  PYTHONPATH=MILP python3 -c "
import sys, finn_milp
finn_milp.RESIDUAL_QUANT_BITS = $JB
sys.argv = ['finn_milp.py'] + '''$COMMON --out-file $OUT/joins$JB/layer_bits_folding_joins$JB.json'''.split()
finn_milp.main()
" > $OUT/joins$JB/solve.log 2>&1 &
done
wait
