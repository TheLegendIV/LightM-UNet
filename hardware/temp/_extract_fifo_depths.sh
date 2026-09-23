#!/bin/sh
proj="$1"
grep -o 'StreamingFIFO_rtl_[0-9]*_[a-z0-9]\{8\}' /tmp/finn_dev_thelegendiv/vivado_stitch_proj_${proj}/make_project.tcl | sort -u > /tmp/fifo_list_${proj}.txt
while read d; do
  base=$(echo "$d" | sed 's/_[a-z0-9]\{8\}$//')
  f=/tmp/finn_dev_thelegendiv/code_gen_ipgen_${d}/${base}.v
  depth=$(grep -m1 -o '\.depth([0-9]*)' "$f" 2>/dev/null)
  echo "$base $depth"
done < /tmp/fifo_list_${proj}.txt
