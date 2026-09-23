#!/bin/bash
grep -rH "Start of session at\|^# Process ID" /tmp/finn_dev_thelegendiv/code_gen_ipgen_*/project_*/sol1/impl/ip/vivado.log 2>/dev/null | grep -B1 "Sep 22 01:1[2-4]" 
echo "---vitis_hls.log start times---"
grep -rH "Starting HLS\|^Vivado(TM) HLS" /tmp/finn_dev_thelegendiv/code_gen_ipgen_*/vitis_hls.log 2>/dev/null | head -40
