#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
D=finn_deployment_outputs/S12_dense_512_u4_analytical_v1_ft15ep_int6_fps250_lat200_milpfold_8way_20261010_050602
ls $D
echo "-- intermediate_models"
ls -la $D/intermediate_models | head -30
echo "-- procs"
ps -eo args | grep -E 'finn_s12_build|vivado|vitis_hls' | grep -v grep | cut -c1-100 | head
echo "-- log tail"
grep -nE 'Completed|ERROR|Traceback|OUTPUT_DIR|rtlsim' /tmp/ooc_S12_512_full.log | tail -15 | cut -c1-200
