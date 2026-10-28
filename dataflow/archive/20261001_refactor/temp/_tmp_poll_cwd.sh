#!/bin/bash
echo ---PS---
ps -eo pid,ppid,lstart,etime,pcpu,rss,cmd | grep -E 'vivado -mode batch|python3.*8way_full_v3' | grep -v grep
echo ---CWD---
for p in $(ps -eo pid,cmd | grep 'vivado -mode batch -source make_project.tcl' | grep -v grep | awk '{print $1}'); do
  echo "pid=$p cwd=$(readlink -f /proc/$p/cwd 2>/dev/null)"
done
echo ---ERRCHECK---
grep -n 'Traceback\|CRITICAL' /tmp/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_and_synth.log | tail -20
