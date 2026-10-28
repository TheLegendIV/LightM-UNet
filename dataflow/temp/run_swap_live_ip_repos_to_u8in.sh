#!/bin/bash
set -e
P=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical
A=$P/archive_pre_20261009_ip_repo_swap_u8in
mkdir -p $A
cp -n $P/s12_256_analytical.xpr $A/
cp -n $P/s12_256_analytical.srcs/sources_1/bd/top/top.bd $A/
source /tools/Xilinx/Vivado/2022.2/settings64.sh
cd /tmp
vivado -mode batch -nojournal -log /tmp/swap_live_ip_repos_to_u8in.log -source /tmp/swap_live_ip_repos_to_u8in.tcl > /dev/null 2>&1 || true
grep -E 'existing=|KEEP|  P[0-9]|new=|ipdefs|SWAP_U8IN_DONE|ERROR|CRITICAL' /tmp/swap_live_ip_repos_to_u8in.log | cut -c1-250
