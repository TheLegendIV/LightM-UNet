#!/bin/bash
P=/home/thelegendiv/finn/vivado_projects/S12_512_bilin_int8/s12_512_bilin_int8
echo "xpr mtime: $(stat -c %y $P/s12_512_bilin_int8.xpr | cut -c1-19)   now: $(date +%T)"
echo "== p0 run entries in xpr"; grep -c 'top_GenericPartition_0_0_0_synth_1' $P/s12_512_bilin_int8.xpr
echo "== p0 run dir"; ls -la --time-style=long-iso $P/s12_512_bilin_int8.runs/top_GenericPartition_0_0_0_synth_1 2>&1 | head
echo "== p0 run log errors (stale?)"; grep -c 'Runs 36-287' $P/s12_512_bilin_int8.runs/top_GenericPartition_0_0_0_synth_1/runme.log 2>&1
echo "== run fileset: pool sources"
grep -c 'StreamingMaxPool' $P/s12_512_bilin_int8.runs/top_GenericPartition_0_0_0_synth_1/*.tcl 2>/dev/null
grep -l 'Pool_hls' $P/s12_512_bilin_int8.runs/top_GenericPartition_0_0_0_synth_1/*.tcl 2>/dev/null
ls $P/s12_512_bilin_int8.gen/sources_1/bd/top/ipshared 2>/dev/null | wc -l
find $P/s12_512_bilin_int8.gen -name 'GenericPartition_0.v' | head -2
