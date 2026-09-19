#!/bin/bash
echo ---PS---
ps -eo pid,ppid,etime,pcpu,rss,cmd | grep -E 'vivado -mode batch|per_partition_synth' | grep -v grep
echo ---LOGTAIL---
tail -15 /tmp/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_and_synth.log
echo ---RESULTS-4-7---
for i in 4 5 6 7; do
  echo "partition $i:"
  cat /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_20260918_144349/report/ooc_synth_partition_$i.json
  echo
done
