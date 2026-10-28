cd /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350 || exit 1
ls report | head -40
echo ---
grep -n -i -E 'rtlsim|N_OUT|PASS|FAIL|deadlock|continuing' build_dataflow.log | head -60
