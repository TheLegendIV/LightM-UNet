#!/bin/bash
# Poll until partition 1/2/3 OOC synth reports exist, then kill the whole build job tree (root PID 584).
OUT=/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_nearest_conv_upsample_256_trained_rtl_mvau_8way_full_v1_256x256_20260926_153851
ROOT_PID=584
LOG=/tmp/kill_on_partitions_123.log

killtree() {
  local pid=$1
  for child in $(pgrep -P "$pid"); do
    killtree "$child"
  done
  kill -9 "$pid" 2>/dev/null
}

echo "$(date): monitor started, watching for partition 1/2/3 reports under $OUT/report" >> "$LOG"

while true; do
  if [ -f "$OUT/report/ooc_synth_partition_1.json" ] && \
     [ -f "$OUT/report/ooc_synth_partition_2.json" ] && \
     [ -f "$OUT/report/ooc_synth_partition_3.json" ]; then
    echo "$(date): all three report files found, killing process tree rooted at PID $ROOT_PID" >> "$LOG"
    if kill -0 "$ROOT_PID" 2>/dev/null; then
      killtree "$ROOT_PID"
      echo "$(date): kill complete" >> "$LOG"
    else
      echo "$(date): root PID $ROOT_PID no longer running, nothing to kill" >> "$LOG"
    fi
    break
  fi
  sleep 30
done
