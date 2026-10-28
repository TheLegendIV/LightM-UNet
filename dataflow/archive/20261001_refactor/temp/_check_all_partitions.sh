#!/bin/bash
date
echo ---DONE/FAILED---
grep -n -E 'partition [0-9]+ done|partition [0-9]+ FAILED|Traceback|All partitions|wrote CSV' /tmp/finn_fifo_depths_only_512x512.log
echo ---PER-PARTITION-DIRS---
for d in /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/GenericPartition_*; do
  n_stitch=$(ls "$d" 2>/dev/null | grep -c vivado_stitch_proj)
  n_fifosim=$(ls "$d" 2>/dev/null | grep -c verilator_fifosim)
  echo "$d: $n_stitch stitch_projs, $n_fifosim fifosim_dirs"
done
echo ---TAIL---
tail -n 8 /tmp/finn_fifo_depths_only_512x512.log
