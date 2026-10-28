#!/bin/bash
L=/tmp/ooc_S12_bilinear_p56_dwfix.log
ls -la --time-style=full-iso $L | cut -c1-120
date
grep -nE "build exit|restored|partition . done|build done|OOC synth done|rtlsim (PASS|DEADLOCK)|FAILED|Traceback" $L | cut -c1-260
tail -4 $L | cut -c1-220
ps aux | grep -E "finn_s12_build|vivado|vitis_hls" | grep -v grep | wc -l
