#!/bin/bash
for d in /tmp/finn_dev_thelegendiv/code_gen_ipgen_StreamingFIFO_rtl_*; do
  v=$(ls "$d"/StreamingFIFO_rtl_*.v 2>/dev/null | head -1)
  if [ -n "$v" ]; then
    dep=$(grep -oP '\.depth\(\K[0-9]+' "$v" | head -1)
    base=$(basename "$d")
    mtime=$(stat -c '%Y' "$d")
    echo "$base DEPTH=$dep MTIME=$mtime"
  fi
done
