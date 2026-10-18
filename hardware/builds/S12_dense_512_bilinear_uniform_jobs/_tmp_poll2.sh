#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
date
for t in int4_cap40_v1 int8_relaxed_v1 int6_cap40_v1_p7fix; do ls -l --time-style=full-iso build_S12_dense_512_bilinear_$t.log | awk '{print $6,$7,$9}'; done
ps -eo pid,etime,pcpu,args --sort=-pcpu | grep -E '[v]ivado|[v]itis|[v]pl|[f]inn_s12_build|[v]erilator|[X]sim|[x]sim' | cut -c1-170 | head -20
nproc; uptime
