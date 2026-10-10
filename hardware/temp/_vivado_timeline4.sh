P=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical
cd "$P" || exit 1
echo "-- ip repo paths loaded by p1 OOC run (build-dir level, with counts)"
grep -a 'Loaded user IP repository' s12_256_analytical.runs/top_GenericPartition_1_0_0_synth_1/runme.log | grep -o -E 'finn_build_tmp/[A-Za-z0-9_]+/GenericPartition_[0-9]' | sort | uniq -c
echo "-- same for top synth_1"
grep -a 'Loaded user IP repository' s12_256_analytical.runs/synth_1/runme.log | grep -o -E 'finn_build_tmp/[A-Za-z0-9_]+/GenericPartition_[0-9]' | sort | uniq -c
echo "-- synth LUT/FF/BRAM per partition (from OOC synth utilization rpt)"
for n in 0 1 2 3 4 5 6 7; do
  f=$(ls s12_256_analytical.runs/top_GenericPartition_${n}_0_0_synth_1/*utilization_synth.rpt 2>/dev/null | head -1)
  echo "p$n: $(grep -a -m1 -E '^\| (CLB|Slice) LUTs' $f | tr -s ' ') | $(grep -a -m1 -E 'RAMB18 ' $f | tr -s ' ')"
done
echo "-- IP VLNV rev in xci (p1)"
grep -a -m3 -E 'ipRepo|COMPONENT_NAME|"Revision"|<spirit:version' s12_256_analytical.srcs/sources_1/bd/top/ip/top_GenericPartition_1_0_0/top_GenericPartition_1_0_0.xci | head -5
echo "-- Old (v2) vs new builds: dirs"
ls -d /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_* 2>/dev/null
