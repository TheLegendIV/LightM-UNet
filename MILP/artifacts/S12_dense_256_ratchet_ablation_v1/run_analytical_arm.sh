#!/bin/bash
# The analytical-flow arm of the ratchet ablation (MILP/analytical/net_fold.py), under the same conditions as the MILP arms: 256x256 nearest-upsample ReLU net, uniform INT6, 250 fps
# (every node <= 400,000 cycles/frame), mvau_wwidth_max 72, ratchet floor 0.33, objective = cheapest fold that fits each block's budget, FIFOs sized by simulation (skip / prefetch / DWC,
# inter-block FIFOs fixed at depth 2).
# Ratchet: net_fold's native block-to-block rule (a block's budget = min(F, max(floor * F, (1 + pct/100) * slowest node of the previous block))) at 25 % (the same allowance as the MILP arm ratchet_25pct; at 4 % + floor 0.33 the block budgets ratchet down to ~136k cycles and the up5 block fails its simulation check: 9.32 cyc/px steady vs 8.3 target);
# the MILP applies the same rule node by node. No latency cap.
# Slow (~20 min: the blocks are verified by cycle simulation twice). Run in lightmunet_dev:  bash MILP/artifacts/S12_dense_256_ratchet_ablation_v1/run_analytical_arm.sh
set -eu
cd /workspace/LightM-UNet
OUT=MILP/artifacts/S12_dense_256_ratchet_ablation_v1/analytical_25pct
python3 MILP/analytical/net_fold.py --config config_12_dense_relu_nearest_upsample_256 --bits 6 --fps 250 --clock-mhz 100 \
  --ratchet-pct 25 --ratchet-floor 0.33 --mvau-wwidth-max 72 --workers 6 --tag analytical_25pct --out-dir $OUT
