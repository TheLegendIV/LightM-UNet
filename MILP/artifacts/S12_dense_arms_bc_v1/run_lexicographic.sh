#!/bin/bash
# Arms B/C bits + MILP folding. Pass 1: accuracy objective, bits free (4,6,8), all board caps 1.0, FPS >= $1, DSR 2.
# Pass 2 (--lexicographic): bits pinned (epsilon 0), LUT/BRAM/DSP capped at pass 1's use, minimize resources,
# same FPS and DSR 2 -> folding only. --mvau-wwidth-max 80 = FINN cfg mvau_wwidth_max (arm B must use the same).
# Usage (in lightmunet_dev): run_lexicographic.sh <target_fps> <out_subdir>   e.g. 250 lex_dsr2 | 305.17 lex_dsr2_fps305
set -u
FPS=${1:-250}; SUB=${2:-lex_dsr2}
cd /workspace/LightM-UNet
OUT=MILP/artifacts/S12_dense_arms_bc_v1/$SUB
mkdir -p $OUT
python3 MILP/finn_milp.py --config config_12_dense_relu_nearest_upsample_wm \
  --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_upsample_wm.json \
  --candidate-bits 4,6,8 --max-lut-fraction 1.0 --max-bram-fraction 1.0 --max-dsp-fraction 1.0 \
  --max-uram-fraction 1.0 --force-dsp --target-fps $FPS --dsr-ratio 2 --mvau-wwidth-max 80 \
  --lexicographic --time-limit 900 --gap-rel 0.005 \
  --out-file $OUT/layer_bits_folding_$SUB.json > $OUT/solve.log 2>&1
