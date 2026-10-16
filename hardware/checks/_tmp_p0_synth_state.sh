#!/bin/bash
P=/home/thelegendiv/finn/vivado_projects/S12_512_bilin_int8/s12_512_bilin_int8
G=$P/s12_512_bilin_int8.gen/sources_1/bd/top
echo "date: $(date)"
echo "== ipshared dirs"; ls -la --time-style=long-iso $G/ipshared | head -20
echo "== ipshared/093b (partition 0 shared sources?)"; ls -la --time-style=long-iso $G/ipshared/093b 2>&1 | head
echo "== 093b/src"; ls $G/ipshared/093b/src 2>&1 | head -20
echo "== 093b/verilog/rtl_ops"; ls $G/ipshared/093b/verilog/rtl_ops 2>&1 | head -20
echo "== p0 ip src children"; ls -la --time-style=long-iso $G/ip/top_GenericPartition_0_0_0/src 2>&1 | head -20
echo "== which ipshared dir holds GenericPartition_0.v ?"
find $G -name GenericPartition_0.v 2>/dev/null
echo "== run dir state"
R=$P/s12_512_bilin_int8.runs
ls -la --time-style=long-iso $R | grep -E 'GenericPartition_0|synth_1|impl_1'
echo "== xpr: p0 sources referenced"
grep -c 'StreamingMaxPool_hls_0' $P/s12_512_bilin_int8.xpr
grep -o 'ipshared/[0-9a-f]*' $P/s12_512_bilin_int8.xpr | sort | uniq -c | head
