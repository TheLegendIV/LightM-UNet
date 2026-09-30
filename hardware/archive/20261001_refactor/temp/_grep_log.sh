#!/bin/bash
F=$(ls -t /tmp/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_*.log | head -1)
echo "LOGFILE=$F"
grep -n 'Traceback\|AssertionError\|ip_path\|component.xml\|Using pre-existing' "$F" | tail -100
