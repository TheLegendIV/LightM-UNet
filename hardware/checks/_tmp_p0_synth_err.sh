#!/bin/bash
L=/home/thelegendiv/finn/vivado_projects/S12_512_bilin_int8/s12_512_bilin_int8/s12_512_bilin_int8.runs/top_GenericPartition_0_0_0_synth_1/runme.log
echo "== counts"; grep -c 'ERROR: \[Runs 36-287\]' $L
echo "== first 6 distinct missing files"
grep 'ERROR: \[Runs 36-287\]' $L | sed -E "s/.*accessible:'([^']*)'.*/\1/" | sort -u | head -12
echo "== critical warnings"
grep -n -A6 'CRITICAL WARNING' $L | head -40
echo "== tail"
tail -12 $L
