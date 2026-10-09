B=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200
cd $B/GenericPartition_0 || exit 1
ls -d */ ; echo ---
ls -l --time-style=+%F_%T rtlsim_single | head -30
echo --- compile.sh; cat rtlsim_single/compile.sh | cut -c1-900
echo --- stitch dir; ls vivado_stitch_proj_*/ | head -30
echo --- widths per partition: top wrapper port declarations
for n in 0 1 2 3 4 5 6 7; do
  d=$B/GenericPartition_$n
  w=$(ls $d/vivado_stitch_proj_*/GenericPartition_${n}_wrapper.v $d/vivado_stitch_proj_*/*_wrapper.v 2>/dev/null | head -1)
  echo "p$n wrapper=$w"
  grep -E 'tdata|tvalid' $w | head -6
done
echo --- results; for n in 0 7; do cat $B/GenericPartition_$n/rtlsim_single/results.txt; done
