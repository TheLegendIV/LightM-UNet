#!/bin/bash
# Ratchet ablation, hardware side: partition-2 OOC builds of the S12 dense nearest-upsample (noconv) ReLU 256x256 net, uniform INT6.
#   one build per DISTINCT MILP folding (arms with an identical folding share a build, see MILP/artifacts/S12_dense_256_ratchet_ablation_v1/arms_to_build.txt)
#   arms_to_build.txt now lists only the five <arm>_simfifo variants (MILP foldings + simulated FIFO lists); the other arms and the FINN auto-fold control (WITH_CONTROL=1 to rebuild it:
#   ratchet_ablation_finn_autofold, no --folding-json; target fps 250, mvau_wwidth_max 72 = the MILP's values) were built before.
# Run INSIDE the FINN container (HOME=/tmp/home_dir) after docker cp-ing the inputs into the flat /home/thelegendiv/finn/notebooks/enet/ dir (README.md "Inputs").
# Usage:  bash run_arms.sh [arm ...]        (default: the arms in arms_to_build.txt)
#         STEP=bridge bash run_arms.sh      only the cheap steps (preamble + bridge dry run + landed-folding gate); no Vivado. Do this first.
set -eo pipefail
cd /home/thelegendiv/finn/notebooks/enet
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1

MODEL=quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep          # onnx basename (QAT'd 15-epoch export); the PTQ one is ..._finn_calibrated
CONV_ORDER=quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_conv_order.json   # shared by both exports
TARGET_FPS=250
WWIDTH=72            # same as the MILP's --mvau-wwidth-max
PART=2               # partition 2 = stage2.0 .. stage2.4 (dilations 2, 4, 8, 16, 2): check against the conv order before trusting the label
MAXJOBS=4
STEP=${STEP:-all}

ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then mapfile -t ARMS < arms_to_build.txt; fi
echo "arms to build: ${ARMS[*]}"

echo "=== preamble ==="
python3 finn_s12_preamble.py "$MODEL" --tag ratchet256 2>&1 | tee /tmp/ratchet256_preamble.log
PDIR=$(ls -d finn_deployment_outputs/ratchet256_preamble_* | tail -1)
echo "preamble dir: $PDIR"

echo "=== bridge dry run + landed-folding gate (no Vivado) ==="
rc=0
for arm in "${ARMS[@]}"; do
  python3 finn_s12_build.py "$PDIR" --tag "$arm" --conv-order "$CONV_ORDER" --folding-json "layer_bits_folding_${arm}.json" \
      --partitions $PART --target-fps $TARGET_FPS --mvau-wwidth-max $WWIDTH --bridge-only 2>&1 | tee "/tmp/ratchet256_bridge_${arm}.log"
  grep -E "FIFO role bridge" "/tmp/ratchet256_bridge_${arm}.log" || echo "WARNING: no FIFO role bridge line for $arm (folding json without FIFO lists?)"
  ODIR=$(grep -oP '(?<=OUTPUT_DIR=).*' "/tmp/ratchet256_bridge_${arm}.log" | tail -1)
  python3 dump_milpfold_landed_partition.py "$PDIR" "$ODIR/hawq_folding_config_partition${PART}.json" $PART "$PDIR/landed_partition${PART}_${arm}.onnx" \
      > "/tmp/ratchet256_landed_${arm}.log" 2>&1 || { echo "$arm: landed dump FAILED (see /tmp/ratchet256_landed_${arm}.log)"; rc=1; continue; }
  echo "--- $arm ---"
  python3 check_milp_vs_landed_folding.py "layer_bits_folding_${arm}.json" "$PDIR/landed_partition${PART}_${arm}.onnx" --conv-order "$CONV_ORDER" || rc=1
done
echo "LANDED CHECK: $([ $rc -eq 0 ] && echo OK || echo MISMATCH)"
[ "$STEP" = "bridge" ] && exit $rc
[ $rc -ne 0 ] && { echo "refusing to queue builds after a landed-folding mismatch (rerun with STEP=force to override)"; [ "$STEP" != "force" ] && exit 1; }

echo "=== queueing partition-$PART OOC builds (max $MAXJOBS concurrent) ==="
launch() {  # tag [extra finn_s12_build.py args...]
  while [ "$(pgrep -f '^python3 finn_s12_build.py' | wc -l)" -ge $MAXJOBS ]; do sleep 60; done
  tag=$1; shift
  nohup python3 finn_s12_build.py "$PDIR" --tag "$tag" --conv-order "$CONV_ORDER" --partitions $PART --target-fps $TARGET_FPS --mvau-wwidth-max $WWIDTH "$@" \
      > "/tmp/ooc_${tag}.log" 2>&1 &
  echo "$(date): LAUNCHED $tag PID=$!"
  sleep 5
}
launch ratchet_ablation_finn_autofold --fifo-autosize rtlsim           # control: FINN's own step_target_fps_parallelization + real largefifo_rtlsim FIFO autosizing (no MILP folding/FIFO forcing)
for arm in "${ARMS[@]}"; do
  extra=(--folding-json "layer_bits_folding_${arm}.json")
  # this arm's json has no inter_block_fifos/intra_block_fifos: needs FINN's real rtlsim autosizer, not the fixed2-everywhere default
  [ "$arm" = "analytical_25pct_finnfifo" ] && extra+=(--fifo-autosize rtlsim)
  launch "$arm" "${extra[@]}"
done
echo "Tail logs: tail -f /tmp/ooc_*.log ; results: finn_deployment_outputs/<tag>_{milpfold,autofold}_partition${PART}_<ts>/report/"
