B=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200
for n in 0 2 6 7; do
  d=$(ls -d $B/GenericPartition_$n/vivado_stitch_proj_*)
  echo "== p$n $d"
  ls -l --time-style=+%F_%T $d/GenericPartition_${n}_wrapper.v $B/GenericPartition_$n/rtlsim_single/VGenericPartition_${n}_wrapper.h | cut -c30-200
  grep -n -E 'module |tdata|STREAM_BITS|AXI_(I|O)BITS' $d/GenericPartition_${n}_wrapper.v | head -14
  grep -c . $d/GenericPartition_${n}_wrapper.v
  grep -E 'tdata' $B/GenericPartition_$n/rtlsim_single/VGenericPartition_${n}_wrapper.h | head -4
done
