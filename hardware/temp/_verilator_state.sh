#!/bin/bash
export HOME=/tmp/home_dir
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
unset VERILATOR_ROOT
echo "--- PATH verilator:"; which -a verilator; verilator --version 2>&1 | head -1
echo "--- pip verilator:"; perl /tmp/home_dir/.local/lib/python3.10/site-packages/verilator/bin/verilator --version 2>&1 | head -1
echo "--- env VERILATOR*:"; env | grep -i verilator
cd /home/thelegendiv/finn
echo "--- finn verilator usage in set_fifo_depths / pyverilator:"
grep -n "verilator" src/finn/transformation/fpgadataflow/set_fifo_depths.py | head -20
grep -n "def verilator_fifosim\|VERILATOR_ROOT\|verilator_bin\|which(" src/finn/util/fpgadataflow.py src/finn/util/pyverilator.py src/finn/util/basic.py 2>/dev/null | head -30
echo "--- running jobs:"; ps aux | grep -E "finn_s12_build|vivado|verilator|vlfifo|Vtop" | grep -v grep | cut -c1-200
