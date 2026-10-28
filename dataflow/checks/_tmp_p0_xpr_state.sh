#!/bin/bash
P=/home/thelegendiv/finn/vivado_projects/S12_512_bilin_int8/s12_512_bilin_int8
echo "now: $(date)"
echo "== file times"
ls -la --time-style=full-iso $P/s12_512_bilin_int8.xpr $P/s12_512_bilin_int8.srcs/sources_1/bd/top/top.bd $P/s12_512_bilin_int8.srcs/sources_1/bd/top/ip/top_GenericPartition_0_0_0/*.xci 2>&1 | cut -c30-200
echo "== vivado processes"
ps -eo pid,etime,args | grep -i -E 'vivado' | grep -v grep | cut -c1-200
echo "== project locks"
ls -la $P/*.lock $P/.* 2>/dev/null | head
echo "== p0 repo lines in xpr"
grep -c 'ipool_p0' $P/s12_512_bilin_int8.xpr
grep -c '_pairs/GenericPartition_0' $P/s12_512_bilin_int8.xpr
echo "== backup xpr (pre swap) p0 lines"
grep -c '_pairs/GenericPartition_0' /home/thelegendiv/finn/vivado_projects/S12_512_bilin_int8/_backup_pre_ipool_p0/s12_512_bilin_int8.xpr
