#!/bin/bash
set -e
P=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical
A=$P/archive_pre_20261009_ip_repo_swap_p1fix
mkdir -p $A
cp -n $P/s12_256_analytical.xpr $A/
cp -n $P/s12_256_analytical.srcs/sources_1/bd/top/top.bd $A/
source /tools/Xilinx/Vivado/2022.2/settings64.sh
cd /tmp
vivado -mode batch -nojournal -log /tmp/swap_live_ip_repos_p1fix.log -source /tmp/swap_live_ip_repos_p1fix.tcl > /dev/null 2>&1 || true
grep -E 'existing=|DROP|  P1|new=|ipdefs|  [a-z.:_A-Z0-9]*GenericPartition_1|SWAP_P1FIX_DONE|ERROR|CRITICAL|WARNING' /tmp/swap_live_ip_repos_p1fix.log | cut -c1-300
echo ---- ip status
sed -n '/report_ip_status/,/SWAP_P1FIX_DONE/p' /tmp/swap_live_ip_repos_p1fix.log | cut -c1-250 | head -60
