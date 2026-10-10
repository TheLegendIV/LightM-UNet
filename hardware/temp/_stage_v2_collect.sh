#!/bin/bash
B=/home/thelegendiv/finn/notebooks/enet
O=$B/finn_deployment_outputs/S12_dense_256_u4_analytical_v2_ft15ep_int6_fps250_lat200_milpfold_8way_20261008_040600
T=$B/finn_build_tmp/S12_dense_256_u4_analytical_v2_ft15ep_int6_fps250_lat200
S=/tmp/v2_stage
rm -rf $S; mkdir -p $S/report $S/build_tmp
cp $O/report/*.json $S/report/
for i in 0 1 2 3 4 5 6 7; do
  for f in $T/GenericPartition_$i/synth_out_of_context_*/results_*_wrapper/vivado.log; do
    [ -f "$f" ] || continue
    rel=${f#$T/}
    mkdir -p $S/build_tmp/$(dirname $rel)
    cp "$f" $S/build_tmp/$rel
  done
done
echo "-- staged logs"; find $S/build_tmp -name vivado.log | sort
echo "-- times"; ls -l --time-style=long-iso $O/build_dataflow.log $O/time_per_step.json $O/report/ooc_synth_and_timing_per_partition.json $O/report/rtlsim_performance.json
echo "-- rtlsim_performance"; cat $O/report/rtlsim_performance.json; echo
echo "-- per-partition rtlsim"; python3 -c "
import json
r=json.load(open('$O/report/rtlsim_per_partition.json'))
for k in sorted(r): print(k, r[k].get('deadlock'), r[k].get('latency_cycles'), r[k].get('N_OUT_TXNS'), r[k].get('expected_out_txns'))
"
echo "-- first/last build log lines"; head -2 $O/build_dataflow.log | cut -c1-120; tail -2 $O/build_dataflow.log | cut -c1-120
