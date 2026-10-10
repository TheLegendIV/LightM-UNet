B=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200/GenericPartition_0
cd "$B" || exit 1
echo "-- p0 threshold files"
ls -d code_gen_ipgen_*Thresholding* 2>/dev/null
for d in $(ls -d code_gen_ipgen_*Thresholding* 2>/dev/null | head -2); do
  ls $d | head -20
  f=$(ls $d/*.dat 2>/dev/null | head -1)
  [ -n "$f" ] && { echo "$f"; wc -l "$f"; head -c 600 "$f"; echo; }
done
echo "-- first-layer thresholds in project synth (readmem path for p0)"
grep -a -o "readmem data file '[^']*'" /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.runs/top_GenericPartition_0_0_0_synth_1/runme.log | sed 's#.*/##' | sort | uniq -c | head
