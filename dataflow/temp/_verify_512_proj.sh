#!/bin/bash
P=/home/thelegendiv/finn/vivado_projects/S12_512_analytical/s12_512_analytical
echo "== 256 refs in xpr/bd (should be 0)"
grep -c "S12_dense_256" $P/s12_512_analytical.xpr
grep -c "S12_dense_256" $P/s12_512_analytical.srcs/sources_1/bd/top/top.bd
echo "== 512 refs in xpr"
grep -c "S12_dense_512" $P/s12_512_analytical.xpr
echo "== stale 256 refs in any xci/xml"
grep -rl "S12_dense_256" $P/s12_512_analytical.srcs 2>/dev/null | head
echo "== validate/gen warnings"
grep -E "^(CRITICAL WARNING|ERROR)|BD 41-|BD 5-" /tmp/vivado_512_make.log | cut -c1-250 | sort | uniq -c | head -20
echo "== ip status (locked/out-of-date)"
grep -iE "locked|out-of-date|upgrade" /tmp/vivado_512_make.log | cut -c1-200 | head
echo "== address map / widths"
ls $P
