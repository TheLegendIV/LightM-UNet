#!/bin/bash
total=0
count=0
# StreamingFIFO_rtl impl_style=rtl wrapper files (Q_srl-based)
for f in /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/GenericPartition_5/code_gen_ipgen_StreamingFIFO_rtl_*/StreamingFIFO_rtl_*.v; do
  d=$(grep -oP '(?<=\.depth\()[0-9]+' "$f" | head -1)
  if [ -n "$d" ]; then
    total=$((total + d))
    count=$((count + 1))
    echo "$(basename "$f") depth=$d"
  fi
done
echo "OLD (pre-fix) GenericPartition_5: count=$count total_depth=$total"
