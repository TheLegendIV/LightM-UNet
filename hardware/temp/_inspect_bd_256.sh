#!/bin/bash
source /tools/Xilinx/Vivado/2022.2/settings64.sh
cd /tmp
rm -f /tmp/inspect_bd_256.log
vivado -mode batch -nojournal -log /tmp/inspect_bd_256.log -source /tmp/_inspect_bd_256.tcl > /dev/null 2>&1 || true
grep -E '^(CELL|PIN|REPO|FILE|===|top|[A-Za-z0-9_]+ (Complete|Not|Out|Queued|Running))|xczu|ERROR|CRITICAL' /tmp/inspect_bd_256.log | cut -c1-260 | head -150
