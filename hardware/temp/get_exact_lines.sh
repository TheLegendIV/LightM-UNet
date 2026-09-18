#!/bin/bash
BD=/tmp/finn_dev_thelegendiv/vivado_stitch_proj_d9ohtyc0/finn_vivado_stitch_proj.srcs/sources_1/bd/GenericPartition_2/GenericPartition_2.bd
for n in 3 6 12 28 34 50 55 56 69 72 77 78 94 100; do
  name="StreamingFIFO_rtl_$n"
  lines=$(grep -n "\"$name\": {" "$BD" | cut -d: -f1)
  for line in $lines; do
    memtype_line=$(sed -n "${line},$((line+60))p" "$BD" | grep -n '"FIFO_MEMORY_TYPE"' | head -1 | cut -d: -f1)
    if [ -n "$memtype_line" ]; then
      abs_memtype_line=$((line + memtype_line - 1))
      value_line=$((abs_memtype_line + 1))
      valuetext=$(sed -n "${value_line}p" "$BD")
      echo "$name: FIFO_MEMORY_TYPE key line=$abs_memtype_line value line=$value_line text=[$valuetext]"
    fi
  done
done
