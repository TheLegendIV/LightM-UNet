#!/bin/bash
BD=/tmp/finn_dev_thelegendiv/vivado_stitch_proj_d9ohtyc0/finn_vivado_stitch_proj.srcs/sources_1/bd/GenericPartition_2/GenericPartition_2.bd
for n in 3 6 12 28 33 34 50 55 56 69 72 77 78 94 100; do
  name="StreamingFIFO_rtl_$n"
  lines=$(grep -n "\"$name\": {" "$BD" | cut -d: -f1)
  found=0
  for line in $lines; do
    window=$(sed -n "${line},$((line+60))p" "$BD")
    depth=$(echo "$window" | grep -A1 '"FIFO_DEPTH"' | grep 'value' | head -1)
    memtype=$(echo "$window" | grep -A1 '"FIFO_MEMORY_TYPE"' | grep 'value' | head -1)
    if [ -n "$memtype" ]; then
      echo "$name (def line $line): depth=[$depth] memtype=[$memtype]"
      found=1
      break
    fi
  done
  if [ "$found" -eq 0 ]; then
    echo "$name: no FIFO_MEMORY_TYPE definition found (candidate lines: $lines)"
  fi
done
