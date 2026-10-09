cd /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350 || exit 1
ls | head -30
for p in 0 1 2 3 4 5 6 7; do
  f=$(ls -d *partition${p}* 2>/dev/null | head -1)
  echo "== p$p $f"
  cat "$f/report/rtlsim_performance.json" 2>/dev/null | tr -d '\n ' | head -c 700
  echo
done
