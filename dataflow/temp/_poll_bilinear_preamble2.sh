#!/bin/bash
date +%H:%M:%S
ps aux | grep -E "finn_s12_preamble" | grep -v grep | cut -c1-120
tail -n 6 /tmp/preamble_bilinear.log | cut -c1-250
cd /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs
d=$(ls -d S12_dense_256_u4_bilinear_analytical_v1_ft15ep_preamble_* | tail -1)
echo "== $d"
ls $d $d/intermediate_models 2>/dev/null | cut -c1-120
