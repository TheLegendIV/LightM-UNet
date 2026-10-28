#!/bin/bash
ps -eo pid,etime,args | grep -E 'finn_s12|vivado|vitis_hls' | grep -v grep | cut -c1-170
echo "--- preamble log tail"
grep -vE "Warning|warn" /tmp/preamble_512.log | tail -${1:-6} | cut -c1-200
