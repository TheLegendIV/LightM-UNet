#!/bin/bash
LOG=/tmp/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_relaunch.log
OUTDIR=/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_512x512_20260922_102900

echo "--- last partition marker before each fifo depth line ---"
awk '
/partition[0-9]/ { last=$0; lastnum=NR }
/rounding-up FIFO depth/ { print "line " NR ": nearest partition marker = (line " lastnum ") " last " -- " $0 }
' "$LOG"

echo ""
echo "--- on-disk fifo/report files for partition1 ---"
find "$OUTDIR" -ipath '*partition1*' 2>/dev/null | grep -iE 'fifo|report|json' 

echo ""
echo "--- any auto_fifo / report json anywhere ---"
find "$OUTDIR" -iname '*fifo*' -o -iname '*report*.json' 2>/dev/null | head -40
