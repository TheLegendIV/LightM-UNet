#!/bin/bash
# The analytical-flow arm of the DSR ablation (MILP/analytical/net_fold.py), under the same conditions as the MILP arms: 256x256 nearest-upsample ReLU net, uniform INT6, 250 fps
# (every node <= 400,000 cycles/frame), mvau_wwidth_max 72, DSR floor 0.33, objective = cheapest fold that fits each block's budget, FIFOs sized by simulation (skip / prefetch / DWC,
# inter-block FIFOs fixed at depth 2).
# DSR: net_fold's native block-to-block rule (a block's budget = min(F, max(floor * F, (1 + pct/100) * slowest node of the previous block))) at 25 % (the same allowance as the MILP arm ratchet_25pct; at 4 % + floor 0.33 the block budgets DSR down to ~136k cycles and the up5 block fails its simulation check: 9.32 cyc/px steady vs 8.3 target);
# the MILP applies the same rule node by node. No latency cap.
# ~5 min (the blocks are verified by cycle simulation in parallel workers, one per distinct block shape). Run in lightmunet_dev:  bash MILP/artifacts/S12_dense_256_ratchet_ablation_v1/run_analytical_arm.sh
set -eu
cd /workspace/LightM-UNet
OUT=MILP/artifacts/S12_dense_256_ratchet_ablation_v1
source $OUT/arms_config.sh                        # CONFIG BITS FPS CLOCK_MHZ DSR_FLOOR WWIDTH: the same values as the MILP arms
python3 MILP/analytical/net_fold.py --config $CONFIG --bits $BITS --fps $FPS --clock-mhz $CLOCK_MHZ   --dsr-pct 25 --dsr-floor $DSR_FLOOR --mvau-wwidth-max $WWIDTH --workers 6 --tag analytical_25pct --out-dir $OUT/analytical_25pct
