#!/bin/bash
X=/home/thelegendiv/finn/vivado_projects/S12_256_bilinear_analytical/s12_256_bilinear_analytical/s12_256_bilinear_analytical.xpr
B=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200
echo "== ip repo entries in xpr"
grep -n -i "ip_repo\|IPRepoPath\|IPUserFiles" $X | cut -c1-200 | head -5
grep -o 'Option Name="IPRepoPath"[^/]*' $X | head
grep -o 'Path="[^"]*GenericPartition_[56][^"]*"' $X | sort -u
grep -c "RepoPath\|ip_repo" $X
echo "== on disk P5/P6 stitch + hls ip dirs"
ls -d --time-style=long-iso -l $B/GenericPartition_5/vivado_stitch_proj_*/ip $B/GenericPartition_6/vivado_stitch_proj_*/ip
for p in 5 6; do ls -d -l --time-style=long-iso $B/GenericPartition_$p/code_gen_ipgen_*_hls_*/*/sol1/impl/ip | awk '{print $6, $7, $8}' | head -20; done
echo "== open vivado?"; ps aux | grep -i "[v]ivado" | wc -l
