P=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical
cd "$P/s12_256_analytical.runs" || exit 1
date
ls -l --time-style=+%F_%T -d * | cut -c30-120
echo "-- OOC partition run dcp times"
for r in top_GenericPartition_*_0_synth_1; do
  f=$(ls $r/*.dcp 2>/dev/null | head -1)
  echo "$r: $(ls -l --time-style=+%F_%T $f 2>/dev/null | cut -c30-100)  status=$(cat $r/.vivado.end.rst >/dev/null 2>&1 && echo ended)"
done
echo "-- source paths seen by partition 1 OOC synth"
grep -o -E '/home/thelegendiv/finn/[A-Za-z0-9_./-]*(ip_|stitched|finn_deployment|finn_build)[A-Za-z0-9_./-]*' top_GenericPartition_1_0_synth_1/runme.log 2>/dev/null | cut -d/ -f1-9 | sort | uniq -c | head -5
echo "-- where the 8 partitions' IPs are packaged from (component.xml dirs in ip repo paths)"
grep -o -E 'S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_[0-9_]+' ../s12_256_analytical.xpr | sort | uniq -c
grep -o -E 'S12_dense_256_u4_[A-Za-z0-9_]*_8way_[0-9_]+' ../s12_256_analytical.xpr | sort | uniq -c
