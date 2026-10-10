#!/bin/bash
source /tools/Xilinx/Vivado/2022.2/settings64.sh
cd /tmp
vivado -mode batch -nojournal -log /tmp/upgrade_live_p1fix.log -source /tmp/upgrade_live_p1fix.tcl > /dev/null 2>&1 || true
grep -E 'bd cells|reset |ERROR|CRITICAL|UPGRADE_P1_DONE|Upgrad' /tmp/upgrade_live_p1fix.log | cut -c1-250 | head -40
echo ---- non up-to-date
grep -E '^\| (top_|GenericPartition|[a-z_A-Z0-9]+) ' /tmp/upgrade_live_p1fix.log | grep -v 'Up-to-date' | cut -c1-200 | head -20
