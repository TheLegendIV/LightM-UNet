#!/bin/bash
source /tools/Xilinx/Vivado/2022.2/settings64.sh
cd /tmp
vivado -mode batch -nojournal -nolog -source /tmp/_tmp_list_ip_repos.tcl 2>&1 | grep -E 'COUNT=|BD_FILES|ERROR' 
