cd /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200/GenericPartition_7 2>/dev/null || exit 1
ls | head -50
echo ---
for d in MVAU_rtl_*; do
  echo "== $d"
  f=$(ls $d/*_wrapper.v 2>/dev/null | head -1)
  grep -nE "parameter\s+(IS_MVU|COMPUTE_CORE|PUMPED_COMPUTE|MW|MH|PE|SIMD|ACTIVATION_WIDTH|WEIGHT_WIDTH|ACCU_WIDTH|NARROW_WEIGHTS|SIGNED_ACTIVATIONS|SEGMENTLEN|FORCE_BEHAVIORAL)\b" $f | sed 's/\t/ /g'
done
echo --- tools
source /tools/Xilinx/Vivado/2022.2/settings64.sh >/dev/null 2>&1
which xvlog verilator; verilator --version
