#!/bin/bash
P=/home/thelegendiv/finn/vivado_projects/S12_512_bilin_int8/s12_512_bilin_int8
C=$P/s12_512_bilin_int8.cache/ip/2022.2
echo "== cache entries"
for d in $C/*/; do
  b=$(basename $d)
  x=$(ls $d | grep -E '\.xci$' | head -1)
  echo "$b $(stat -c %y $d | cut -c1-16) xci=$x $(ls $d | tr '\n' ' ' | cut -c1-120)"
done
echo "== xci module names of cache entries mentioning GenericPartition"
grep -l 'GenericPartition' $C/*/*.xci 2>/dev/null
echo "== partition 0 ip dir in the project"
ls -la --time-style=long-iso $P/s12_512_bilin_int8.srcs/sources_1/bd/top/ip/top_GenericPartition_0_0_0 | head -20
echo "== OOC run dirs / status for partition 0 and 1"
ls -d $P/s12_512_bilin_int8.runs/top_GenericPartition_*_synth_1
cat $P/s12_512_bilin_int8.runs/top_GenericPartition_1_0_0_synth_1/.vivado.end.rst 2>/dev/null | head -2
echo "== global ip cache locations"
ls -la ~/.Xilinx 2>/dev/null | head; ls /tmp/home_dir/.Xilinx 2>/dev/null | head
