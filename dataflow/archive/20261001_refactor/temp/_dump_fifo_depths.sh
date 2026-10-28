#!/bin/bash
for d in /tmp/finn_dev_*/code_gen_ipgen_*StreamingFIFO_rtl*; do
  v=$(ls "$d"/*.v 2>/dev/null | grep -v fifo_template | head -1)
  if [ -n "$v" ]; then
    dep=$(grep -oP 'depth\s*=\s*[0-9]+|\.depth\([0-9]+\)' "$v" | grep -oP '[0-9]+' | head -1)
    base=$(basename "$d")
    echo "$base DEPTH=$dep"
  fi
done
