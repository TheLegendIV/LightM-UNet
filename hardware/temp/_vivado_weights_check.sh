P=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical
cd "$P/s12_256_analytical.runs" || exit 1
echo "-- readmem / init-file / ROM warnings per partition OOC synth"
for n in 0 1 2 3 4 5 6 7; do
  l=top_GenericPartition_${n}_0_0_synth_1/runme.log
  echo "p$n: readmem=$(grep -a -c -i 'readmem\|cannot open\|Failed to open\|can not open' $l)  unconnected/undriven=$(grep -a -c -E 'Synth 8-(3295|3331|3332|6014|7129|3917|3936)' $l)  const0=$(grep -a -c 'Synth 8-3917\|Synth 8-3936' $l)"
done
echo "-- sample readmem related lines (p1)"
grep -a -i -m8 'readmem\|cannot open\|Failed to open\|\.dat' top_GenericPartition_1_0_0_synth_1/runme.log | cut -c1-220
echo "-- top synth: critical/error/blackbox"
grep -a -c -E 'CRITICAL WARNING|ERROR' synth_1/runme.log
grep -a -i -E 'black ?box|Synth 8-3[0-9]{3}.*(not found|unresolved)' synth_1/runme.log | head -5 | cut -c1-200
echo "-- impl: critical warnings, timing"
grep -a -c 'CRITICAL WARNING' impl_1/runme.log
grep -a -i -E 'WNS|Timing constraints are not met|timing_summary' impl_1/runme.log | tail -4 | cut -c1-200
echo "-- axi stream width links around IODMA in the BD (p0 in, p7 out) from the saved xsa hwh"
cd "$P"
ls -l --time-style=+%F_%T *.hwh s12_256_analytical.gen/sources_1/bd/top/hw_handoff/*.hwh 2>/dev/null | cut -c30-170
H=$(ls s12_256_analytical.gen/sources_1/bd/top/hw_handoff/top.hwh 2>/dev/null)
[ -n "$H" ] && grep -a -o -E 'NAME="(p0_input_IODMA_hls_0|p7_output_IODMA_hls_0|GenericPartition_0|GenericPartition_7)"[^>]*|PARAMETER NAME="(TDATA_NUM_BYTES|C_S_AXIS[A-Z_]*|C_M_AXIS[A-Z_]*)" VALUE="[0-9a-fx]*"' $H | head -20
