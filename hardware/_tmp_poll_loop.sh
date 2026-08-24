#!/bin/bash
LOG="/home/thelegendiv/finn/notebooks/enet/_poll_status.log"
BUILD_LOG="/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/stitched_ip_partitioned8way_quantEnet_s19_double_mid_int8_largefifo_rtlsim_20260820_101224/build_dataflow.log"
while true; do
  RESUME_LOG=$(ls -t /tmp/resume_combine_partitions*.log 2>/dev/null | head -1)
  {
    echo "=== $(date) ==="
    echo "--- procs ---"
    ps aux | grep -E 'python3 finn_enet_ip_resume|vivado -mode batch|vitis_hls|verilator' | grep -v grep
    echo "--- build_dataflow.log tail ---"
    tail -5 "$BUILD_LOG" 2>/dev/null
    echo "--- resume log ($RESUME_LOG) tail ---"
    tail -8 "$RESUME_LOG" 2>/dev/null
    echo
  } >> "$LOG" 2>&1
  sleep 600
done

