#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
for t in int4_cap40_v1 int8_relaxed_v1 int6_cap40_v1_p7fix; do
  echo "== $t"
  grep -E 'per-partition rtlsim|partition [0-9] done|Traceback|Error|Done\.' build_S12_dense_512_bilinear_$t.log | tail -12
  tail -c 300 build_S12_dense_512_bilinear_$t.log | tail -2
done
