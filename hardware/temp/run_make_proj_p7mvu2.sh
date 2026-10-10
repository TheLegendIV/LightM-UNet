#!/bin/bash
source /tools/Xilinx/Vivado/2022.2/settings64.sh
cd /tmp
rm -f /tmp/make_proj_p7mvu2.log
vivado -mode batch -nojournal -log /tmp/make_proj_p7mvu2.log -source /tmp/make_proj_p7mvu2.tcl > /dev/null 2>&1 || true
grep -E 'new project|existing=|DROP|  P7|new=|GenericPartition_7|MAKE_PROJ_P7MVU2_DONE|ERROR|CRITICAL|locked|upgrade' /tmp/make_proj_p7mvu2.log | cut -c1-300
echo ---- ip status / runs
sed -n '/=== report_ip_status/,/MAKE_PROJ_P7MVU2_DONE/p' /tmp/make_proj_p7mvu2.log | cut -c1-250 | head -60
