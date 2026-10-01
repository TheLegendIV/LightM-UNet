#!/bin/bash
# Partition-2-only OOC builds for arms B / C / C-noDSR of the MILP-vs-FINN-auto-fold experiment.
# Same bits everywhere (lex_dsr2_fps305, QAT checkpoint re-quantized); only the folding differs:
#   armB_finn_autofold : FINN step_target_fps_parallelization only (target 305.17, mvau_wwidth_max 80)
#   armC_dsr2          : MILP lex pass 2 folding, --dsr-ratio 2
#   armC_nodsr         : MILP lex pass 2 folding, no DSR
# Run INSIDE the FINN container (HOME=/tmp/home_dir), after docker cp-ing into the flat
# /home/thelegendiv/finn/notebooks/enet/ dir (see README.md "Inputs"). No Vivado until the last step.
set -eo pipefail
cd /home/thelegendiv/finn/notebooks/enet
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
export ARMS_TARGET_FPS=305.17
TAG=lex_dsr2_fps305

echo "=== preamble ==="
python3 finn_hawq_preamble_trained.py "$TAG" 2>&1 | tee /tmp/arms_preamble.log
PDIR=$(grep -oP '(?<=^OUTPUT_DIR= ).*' /tmp/arms_preamble.log | tail -1)
echo "preamble dir: $PDIR"

echo "=== bridge: MILP foldings -> FINN configs (one preamble, two outputs) ==="
python3 finn_hawq_folding_bridge_nearest_upsample.py "$PDIR" layer_bits_folding_lex_dsr2_fps305.json "$PDIR/hawq_folding_config_armC_dsr2.json" 2>&1 | tee /tmp/arms_bridge_dsr2.log
python3 finn_hawq_folding_bridge_nearest_upsample.py "$PDIR" layer_bits_folding_lex_nodsr_fps305.json "$PDIR/hawq_folding_config_armC_nodsr.json" 2>&1 | tee /tmp/arms_bridge_nodsr.log

echo "=== gate: MILP folding landed unchanged (C arms) ==="
bash check_arms_landed.sh "$PDIR" 2>&1 | tee /tmp/arms_landed_check.log

echo "=== arm B, all 8 partitions: FINN auto-fold configs + graphs (for whole-network pricing; no Vivado) ==="
python3 dump_autofold_config_all_partitions.py "$PDIR" armB_finn_autofold ./autofold_armB_all_partitions 2>&1 | tee /tmp/arms_autofold_all.log

echo "=== queueing 3 partition-2 OOC syntheses behind run_queue.sh (max 4 concurrent) ==="
# finn_ooc_partition2_trained.py ignores ARMS_TARGET_FPS; it needs the explicit flags.
FPS_ARGS="--target-fps $ARMS_TARGET_FPS --mvau-wwidth-max 80"
while pgrep -x -f "bash run_queue.sh" > /dev/null; do sleep 60; done
launch() {  # tag [folding_config]
  while [ "$(pgrep -f '^python3 finn_ooc_partition2_trained.py' | wc -l)" -ge 4 ]; do sleep 60; done
  nohup python3 finn_ooc_partition2_trained.py "$PDIR" "$@" $FPS_ARGS > "/tmp/ooc_$1.log" 2>&1 &
  echo "$(date): LAUNCHED $1 PID=$!"
  sleep 5
}
launch armB_finn_autofold
launch armC_dsr2 "$PDIR/hawq_folding_config_armC_dsr2.json"
launch armC_nodsr "$PDIR/hawq_folding_config_armC_nodsr.json"
echo "Tail logs: tail -f /tmp/ooc_arm*.log"
