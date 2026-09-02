#!/bin/bash
echo "---vivado/sim procs---"
ps aux | grep -iE 'vivado|vsim|xsim|xelab|vitis' | grep -v grep
echo "---python3 procs---"
ps aux | grep python3 | grep -v grep
echo "---log tail---"
tail -20 /tmp/hawq_8_2_w20_acc2x_dummy_8way_full.log
echo "---errors so far---"
grep -niE 'error|traceback|assert' /tmp/hawq_8_2_w20_acc2x_dummy_8way_full.log | tail -20
