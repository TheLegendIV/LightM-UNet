#!/bin/bash
E=/home/thelegendiv/finn/notebooks/enet
echo "== p7mvu2 log tail"; grep -nE "OOC synth done|rtlsim|Traceback|stitch|DONE|PASS|FAIL" /tmp/p7mvu2_build.log | cut -c1-300 | tail -15
echo "== p7mvu2 build dir"
B=$E/finn_build_tmp/S12_dense_256_u4_u8in_int6_p7mvu2/GenericPartition_7
ls -d $B/vivado_stitch_proj_*/ip
ls -d $B/code_gen_ipgen_*_hls_*/*/sol1/impl/ip | wc -l
ls $B/vivado_stitch_proj_*/ip | head
echo "== p7mvu build dir"
B1=$E/finn_build_tmp/S12_dense_256_u4_u8in_int6_p7mvu/GenericPartition_7
ls -d $B1/vivado_stitch_proj_*/ip
echo "== live project currently-registered P7 repos"
P=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical
grep -a 'GenericPartition_7' $P/s12_256_analytical.xpr | grep IPRepoPath | sed 's#.*finn_build_tmp/##' | cut -c1-220
echo "== other-partition build tags in live xpr"
grep -a IPRepoPath $P/s12_256_analytical.xpr | sed -E 's#.*finn_build_tmp/([^/]+)/(GenericPartition_[0-9]).*#\1 \2#' | sort | uniq -c
echo "== bd P7 cell ref"
grep -a -n 'GenericPartition_7' $P/s12_256_analytical.srcs/sources_1/bd/top/top.bd | head
echo "== p7 port compat: stitched component.xml bus/ports"
grep -a -o 'spirit:name>[a-z_]*axis_0[^<]*' $B/vivado_stitch_proj_*/ip/component.xml | sort -u | head
grep -a -o 'spirit:name>[a-z_]*axis_0[^<]*' $B1/vivado_stitch_proj_*/ip/component.xml | sort -u | head
