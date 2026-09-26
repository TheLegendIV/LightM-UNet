#!/bin/bash
echo "=== per-partition report/config JSON files across all nouram_512x512 attempts ==="
find ~/finn/notebooks/enet/finn_deployment_outputs -iname "*512x512*" -maxdepth 1 -type d
echo
for d in ~/finn/notebooks/enet/finn_deployment_outputs/*512x512*; do
  echo "--- $d ---"
  find "$d" -iname "*fifo*" -o -iname "final_hw_config.json" -o -iname "auto_folding_config.json" 2>/dev/null
done
echo
echo "=== ALL 'rounding-up FIFO depth' lines across ALL relaunch/build logs in /tmp ==="
grep -h "rounding-up FIFO depth" /tmp/*.log 2>/dev/null | sort | uniq -c | sort -rn
echo
echo "=== which log files have FIFO depth info ==="
grep -l "rounding-up FIFO depth\|StreamingFIFO_rtl" /tmp/*.log 2>/dev/null
echo
echo "=== count of StreamingFIFO depth-set lines (not just rounding) per log ==="
for f in /tmp/*.log; do
  c=$(grep -c "StreamingFIFO_rtl" "$f" 2>/dev/null)
  echo "$f: $c"
done
