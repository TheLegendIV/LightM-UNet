#!/bin/bash
D=/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200_milpfold_8way_20261009_225105
echo "== report files"; ls -la $D/report
echo "== rtlsim jsons"
for f in $D/report/rtlsim*.json; do echo "--- $f"; head -c 800 $f; echo; done
echo "== log: deadlock/stall/fail/pass"
grep -niE "deadlock|stall|timeout|timed out|rtlsim.*(pass|fail|ok)|PASS|FAIL|mismatch" $D/build_dataflow.log | cut -c1-250 | head -60
echo "== tail of log"; tail -15 $D/build_dataflow.log | cut -c1-250
