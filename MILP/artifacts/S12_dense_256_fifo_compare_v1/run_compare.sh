#!/bin/bash
# FIFO sizes BEFORE simulation: analytical flow vs MILP under the same target. 256x256 S12 dense nearest-upsample (noconv) ReLU net, uniform INT6, 250 fps (every node <= 400,000 cycles/frame),
# NO --mvau-wwidth-max, both flows on their default rate rule (--dsr-pct 4 --dsr-floor 0.6), no latency cap, MILP objective --min-resources with FIFO modelling (default).
#   milp/layer_bits_folding_milp.json   the MILP solve; its intra_block_fifos / dwcs are the closed-form estimates
#   the analytical estimates need no run: summarize_compare.py rebuilds the analytical folding with net_fold.assemble (no verification) and reads the block models' own FIFO estimates
# Run in lightmunet_dev:  bash MILP/artifacts/S12_dense_256_fifo_compare_v1/run_compare.sh ; python3 MILP/artifacts/S12_dense_256_fifo_compare_v1/summarize_compare.py
set -u
cd /workspace/LightM-UNet
OUT=MILP/artifacts/S12_dense_256_fifo_compare_v1
mkdir -p $OUT/milp
python3 MILP/finn_milp.py --config config_12_dense_relu_nearest_upsample_256 \
  --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_conv_upsample_256.json \
  --candidate-bits 6 --max-lut-fraction 1.0 --max-bram-fraction 1.0 --max-dsp-fraction 1.0 --max-uram-fraction 1.0 \
  --force-dsp --target-fps 250 --min-resources --time-limit 600 --gap-rel 0.001 \
  --out-file $OUT/milp/layer_bits_folding_milp.json > $OUT/milp/solve.log 2>&1
