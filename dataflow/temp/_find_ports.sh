B=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200
for n in 0 1 2 3 4 5 6 7; do
  d=$B/GenericPartition_$n/rtlsim_single
  echo "== p$n"; grep -E 's_axis_0_t(data|valid|ready)|m_axis_0_t(data|valid|ready)|ap_clk|ap_rst_n' $d/VGenericPartition_${n}_wrapper.h | tr -s ' ' | head -8
  cat $d/results.txt | tr '\n' ' '; echo
  grep -a 'Elapsed since' $d/run.log | tail -2 | tr '\n' ' '; echo
done
ls $B/GenericPartition_1/ | grep -v code_gen
nproc; free -g | head -2
