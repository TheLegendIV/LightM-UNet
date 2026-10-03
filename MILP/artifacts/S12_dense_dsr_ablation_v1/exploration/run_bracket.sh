#!/bin/bash
# Bracket the minimum feasible DSR: INT4 uniform (--candidate-bits 4), 100% board caps, 250 FPS floor,
# --min-resources tie-break. Run in lightmunet_dev. Usage: run_bracket.sh <dsr> [<dsr> ...]   ("off" = no DSR)
set -u
# NOTE: exploration/ holds the bracket runs and the 250 FPS variants (incl. 10x); run_bracket.sh there re-finds r_min.
cd /workspace/LightM-UNet
OUT=MILP/artifacts/S12_dense_dsr_ablation_v1
COMMON="--config config_12_dense_relu_nearest_upsample_wm \
  --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_upsample_wm.json \
  --candidate-bits 4 --max-lut-fraction 1.0 --max-bram-fraction 1.0 --max-dsp-fraction 1.0 \
  --max-uram-fraction 1.0 --force-dsp --target-fps 250 --min-resources --time-limit 600 --gap-rel 0.02"
for r in "$@"; do
  tag=bracket_dsr$r; extra="--dsr-ratio $r"; [ "$r" = off ] && { tag=bracket_dsroff; extra=""; }
  mkdir -p $OUT/$tag
  python3 MILP/finn_milp.py $COMMON $extra --out-file $OUT/$tag/layer_bits_folding_$tag.json > $OUT/$tag/solve.log 2>&1 &
done
wait
