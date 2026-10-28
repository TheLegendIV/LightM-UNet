P=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical
cd "$P" || exit 1
echo "-- which build dirs are in ip_repo_paths (xpr)"
grep -o -E 'S12_dense[A-Za-z0-9_]*[0-9]{8}_[0-9]{6}' s12_256_analytical.xpr | sed -E 's#_partition[0-9]+.*##' | sort | uniq -c
echo "-- build-dir stamps referenced by partition 1 generated/synth sources"
for d in s12_256_analytical.gen/sources_1/bd/top/ip/top_GenericPartition_1_0_0 s12_256_analytical.runs/top_GenericPartition_1_0_0_synth_1; do
  echo "$d"
  grep -rhoa -E '[0-9]{8}_[0-9]{6}' "$d" 2>/dev/null | sort | uniq -c | sort -rn | head -5
done
echo "-- partition 1 OOC synth: synth_design source files (first 6 lines mentioning /ip or .v from build dirs)"
grep -a -E 'finn_build_tmp|finn_deployment_outputs|vivado_stitch' s12_256_analytical.runs/top_GenericPartition_1_0_0_synth_1/runme.log | head -6 | cut -c1-200
echo "-- component.xml mtime for each partition in the new build"
B=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200
ls -l --time-style=+%F_%T $B/GenericPartition_*/stitched_ip/ip/component.xml 2>/dev/null | cut -c30-170
echo "-- mtime of generated sources for each partition cell in the project"
for n in 0 1 2 3 4 5 6 7; do
  echo "p$n newest src: $(find s12_256_analytical.gen/sources_1/bd/top/ip/top_GenericPartition_${n}_0_0 -type f -newermt '2026-10-09 00:00' -printf '%TT\n' 2>/dev/null | sort | tail -1)  oldest: $(find s12_256_analytical.gen/sources_1/bd/top/ip/top_GenericPartition_${n}_0_0 -type f -printf '%TF_%TT\n' 2>/dev/null | sort | head -1)"
done
