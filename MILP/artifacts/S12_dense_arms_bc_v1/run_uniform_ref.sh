#!/bin/bash
# Uniform INT4/6/8 reference under the SAME constraints as arm C (caps 1.0, FPS >= 305.17, width cap 80),
# with and without DSR 2; objective --min-resources (bits are fixed by the single --candidate-bits value).
# Usage (in lightmunet_dev): run_uniform_ref.sh
set -u
cd /workspace/LightM-UNet
OUT=MILP/artifacts/S12_dense_arms_bc_v1/uniform_ref
COMMON="--config config_12_dense_relu_nearest_upsample_wm \
  --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_upsample_wm.json \
  --max-lut-fraction 1.0 --max-bram-fraction 1.0 --max-dsp-fraction 1.0 --max-uram-fraction 1.0 \
  --force-dsp --target-fps 305.17 --mvau-wwidth-max 80 --min-resources --time-limit 900 --gap-rel 0.005"
for b in 4 6 8; do
  for d in dsr2 nodsr; do
    tag=int${b}_$d; extra=""; [ $d = dsr2 ] && extra="--dsr-ratio 2"
    mkdir -p $OUT/$tag
    python3 MILP/finn_milp.py $COMMON --candidate-bits $b $extra --out-file $OUT/$tag/layer_bits_folding_$tag.json > $OUT/$tag/solve.log 2>&1 &
  done
done
wait
