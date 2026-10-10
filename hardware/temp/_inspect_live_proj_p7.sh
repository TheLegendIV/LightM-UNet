#!/bin/bash
E=/home/thelegendiv/finn/notebooks/enet
P=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical
echo "== proj dir"; ls -la $P
echo "== xpr ip repo paths"
grep -a -o 'IPRepoPath[^>]*>' $P/s12_256_analytical.xpr | head; 
grep -a -n 'ip_repo\|IPRepo\|RepoPath' $P/s12_256_analytical.xpr | head -40
echo "== BD file GenericPartition cells"
grep -a -o '"GenericPartition_[0-9_]*"[^{]*' $P/s12_256_analytical.srcs/sources_1/bd/top/top.bd | sort -u | head -20
grep -a -o 'GenericPartition_7[^"]*' $P/s12_256_analytical.srcs/sources_1/bd/top/top.bd | sort -u | head
echo "== p7 builds"
for d in S12_dense_256_u4_u8in_int6_p7mvu2 S12_dense_256_u4_u8in_int6_p7mvu S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix S12_dense_256_u4_analytical_v2_ft15ep_int6_fps250_lat200; do
  echo "-- $d"; ls -ld --time-style=long-iso $E/finn_build_tmp/$d/GenericPartition_7 ; ls $E/finn_build_tmp/$d/GenericPartition_7 | grep -i 'stitch\|IODMA' ; ls $E/finn_build_tmp/$d | head -20
done
echo "== combined stitch proj (current)"
ls -d $E/finn_build_tmp/*/combined_stitch_proj_* 2>/dev/null
