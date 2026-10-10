cd /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_p7mvu/GenericPartition_7
for d in code_gen_ipgen_*MVAU_rtl_*; do
  echo "== $d"; ls $d | head -20
  grep -E "parameter (VERSION|COMPUTE_CORE)|\.VERSION|\.COMPUTE_CORE" $d/*_wrapper.v | head
done
ps aux | grep -E "vivado|vitis_hls|finn_s12" | grep -v grep | wc -l
tail -c 400 /tmp/p7mvu_build.log
