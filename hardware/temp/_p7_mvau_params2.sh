cd /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200/GenericPartition_7
for d in code_gen_ipgen_GenericPartition_7_MVAU_rtl_*; do
  echo "== $d"
  ls $d | head
  f=$(ls $d/*_wrapper.v | head -1)
  grep -nE "parameter\s+(IS_MVU|COMPUTE_CORE|PUMPED_COMPUTE|MW|MH|PE|SIMD|ACTIVATION_WIDTH|WEIGHT_WIDTH|ACCU_WIDTH|NARROW_WEIGHTS|SIGNED_ACTIVATIONS|SEGMENTLEN|FORCE_BEHAVIORAL)\b" $f | sed 's/\t/ /g'
done
