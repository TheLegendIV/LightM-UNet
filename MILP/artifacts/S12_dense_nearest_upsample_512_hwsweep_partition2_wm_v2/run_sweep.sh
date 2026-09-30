#!/bin/bash
# Re-run of the S12_dense_nearest_upsample_512_hwsweep_partition2_wm sweep with a hard 250 FPS
# target (--target-fps 250: every node <= 400000 cycles @100 MHz) and board-only caps (all 1.0).
# Run inside lightmunet_dev from the repo root. Tags mirror the original _wm sweep.
set -u
cd /workspace/LightM-UNet
OUT=MILP/artifacts/S12_dense_nearest_upsample_512_hwsweep_partition2_wm_v2
COMMON="--config config_12_dense_relu_nearest_upsample_wm \
  --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_upsample_wm.json \
  --candidate-bits 4,6,8 --hard-lut-fraction 1.0 --hard-bram-fraction 1.0 --hard-dsp-fraction 1.0 \
  --hard-uram-fraction 1.0 --force-dsp --target-fps 250 --time-limit 600 --gap-rel 0.02"
run() { # tag extra-flags...
  tag=$1; shift
  mkdir -p $OUT/$tag
  python3 MILP/finn_milp.py $COMMON "$@" --out-file $OUT/$tag/layer_bits_folding_$tag.json > $OUT/$tag/solve.log 2>&1 &
}
run baseline_both_off
run dsrSweep_pbiOff_dsr1.5 --dsr-ratio 1.5
run dsrSweep_pbiOff_dsr3.0 --dsr-ratio 3.0
run dsrSweep_pbiOff_dsr7.5 --dsr-ratio 7.5
run dsrSweep_pbiOff_dsr15.0 --dsr-ratio 15.0
run dsrSweep_pbiOff_dsr150.0 --dsr-ratio 150.0
run pbiSweep_dsrOff_pbi1.5 --pbi-ratio 1.5
wait
