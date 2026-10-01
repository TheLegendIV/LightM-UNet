#!/usr/bin/env bash
# Waits for any currently-running finn_ooc_partition2_trained.py job to exit,
# then launches the next queued job, one at a time, until the queue is empty.
# Run detached inside the FINN container:
#   nohup bash run_queue.sh > /tmp/ooc_queue.log 2>&1 &
set -u
cd /home/thelegendiv/finn/notebooks/enet
source /tools/Xilinx/Vivado/2022.2/settings64.sh

PREAMBLE_8X="finn_deployment_outputs/S12_dense_nearest_upsample_512_hwsweep_wm_dsrmin_8x_preamble_20260930_173244"
PREAMBLE_DSROFF="finn_deployment_outputs/S12_dense_nearest_upsample_512_hwsweep_wm_dsr_off_preamble_20260930_170855"

# queue entries: "tag|preamble_dir|folding_config_or_empty|extra_args"
QUEUE=(
  "dsrmin_8x|${PREAMBLE_8X}|${PREAMBLE_8X}/hawq_folding_config_partition2.json|"
  "dsr_ablation_autofold_control|${PREAMBLE_DSROFF}||--target-fps 305.17 --mvau-wwidth-max 80"
)

wait_for_free_slot() {
  while true; do
    # Anchored to the real worker process only -- the detached launcher's own
    # "bash -c ... nohup python3 ..." wrapper also matches an unanchored
    # `-f finn_ooc_partition2_trained.py` grep and stays alive for the whole
    # build (2 PIDs/build), which silently drops the effective cap to ~1
    # concurrent build instead of 4 (fixed 2026-10-01).
    n_running=$(pgrep -f '^python3 finn_ooc_partition2_trained.py' | wc -l)
    if [ "$n_running" -lt 4 ]; then
      return
    fi
    sleep 30
  done
}

for entry in "${QUEUE[@]}"; do
  IFS='|' read -r tag preamble folding extra <<< "$entry"
  wait_for_free_slot
  echo "$(date): launching $tag (slot free, $(pgrep -f '^python3 finn_ooc_partition2_trained.py' | wc -l) currently running)"
  if [ -n "$folding" ]; then
    nohup python3 finn_ooc_partition2_trained.py "$preamble" "$tag" "$folding" $extra > "/tmp/ooc_${tag}.log" 2>&1 &
  else
    nohup python3 finn_ooc_partition2_trained.py "$preamble" "$tag" $extra > "/tmp/ooc_${tag}.log" 2>&1 &
  fi
  disown
  echo "$(date): started $tag pid=$!"
  sleep 5
done

echo "$(date): queue empty, all jobs launched"
