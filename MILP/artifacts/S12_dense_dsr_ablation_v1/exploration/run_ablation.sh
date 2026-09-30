#!/bin/bash
# DSR ablation: INT4 uniform, 100% board caps, 250 FPS floor, --min-resources tie-break (all 6 identical except DSR).
# r_min bracket from run_bracket.sh: infeasible at 1.01, feasible at 1.02 -> r_ref = 1.02; multiples 1/2/4/8/10x.
# Run in lightmunet_dev.
set -u
cd /workspace/LightM-UNet
OUT=MILP/artifacts/S12_dense_dsr_ablation_v1
COMMON="--config config_12_dense_relu_nearest_upsample_wm \
  --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_upsample_wm.json \
  --candidate-bits 4 --hard-lut-fraction 1.0 --hard-bram-fraction 1.0 --hard-dsp-fraction 1.0 \
  --hard-uram-fraction 1.0 --force-dsp --target-fps 250 --min-resources --time-limit 600 --gap-rel 0.001"
run() { # tag extra-flags...
  tag=$1; shift
  mkdir -p $OUT/$tag
  python3 MILP/finn_milp.py $COMMON "$@" --out-file $OUT/$tag/layer_bits_folding_$tag.json > $OUT/$tag/solve.log 2>&1 &
}
run dsrmin_1x   --dsr-ratio 1.02
run dsrmin_2x   --dsr-ratio 2.04
run dsrmin_4x   --dsr-ratio 4.08
run dsrmin_8x   --dsr-ratio 8.16
run dsrmin_10x  --dsr-ratio 10.2
run dsr_off
run dsr_off_fps305 --target-fps 305.1  # extra control: same FPS as the DSR runs (fixed initial.pool floor = 305.17)
wait
