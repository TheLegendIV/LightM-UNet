#!/bin/bash
BD=/tmp/finn_dev_thelegendiv/vivado_stitch_proj_d9ohtyc0/finn_vivado_stitch_proj.srcs/sources_1/bd/GenericPartition_2/GenericPartition_2.bd
for n in 3 6 12 28 33 34 50 55 56 69 72 77 78 94 100; do
  name="StreamingFIFO_rtl_$n"
  line=$(grep -n "\"$name\": {" "$BD" | head -1 | cut -d: -f1)
  if [ -z "$line" ]; then
    echo "$name: NOT FOUND in bd"
    continue
  fi
  # print a window after the cell name line to find FIFO_MEMORY_TYPE and FIFO_DEPTH
  window=$(sed -n "${line},$((line+40))p" "$BD")
  depth=$(echo "$window" | grep -A1 '"FIFO_DEPTH"' | grep 'value' | head -1)
  memtype=$(echo "$window" | grep -A1 '"FIFO_MEMORY_TYPE"' | grep 'value' | head -1)
  echo "$name (line $line): depth=[$depth] memtype=[$memtype]"
done
