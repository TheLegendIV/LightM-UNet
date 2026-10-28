cd /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs
for d in S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350 S12_dense_256_u4_u8in_int6_p7mvu_milpfold_probe7_*; do
  echo "== $d"; cat $d/report/ooc_synth_partition_7.json 2>/dev/null | tr -d '\n ' | cut -c1-600; echo
  cat $d/report/rtlsim_partition_7.json 2>/dev/null | tr -d '\n ' | cut -c1-300; echo
done
