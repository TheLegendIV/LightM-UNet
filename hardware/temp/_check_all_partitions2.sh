#!/bin/bash
date
echo ---ALL-PARTITION-DIRS---
ls -d /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/GenericPartition_* 2>/dev/null
echo ---PER-PARTITION-DETAIL---
for d in /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/GenericPartition_*; do
  n_ipgen=$(ls "$d" 2>/dev/null | grep -c code_gen_ipgen)
  n_stitch=$(ls "$d" 2>/dev/null | grep -c vivado_stitch_proj)
  n_fifosim=$(ls "$d" 2>/dev/null | grep -c verilator_fifosim)
  echo "$d : ipgen=$n_ipgen stitch=$n_stitch fifosim=$n_fifosim"
done
echo ---DONE/FAILED-MARKERS---
grep -n -E 'partition [0-9]+ done|partition [0-9]+ FAILED|Traceback|All partitions' /tmp/finn_fifo_depths_only_512x512.log
