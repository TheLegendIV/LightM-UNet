#!/bin/bash
export HOME=/tmp/home_dir
cd /home/thelegendiv/finn/notebooks/enet
nohup python3 -u _bisect_preamble_models.py finn_deployment_outputs/u8in_signbias2_preamble_20261008_174934 quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep_u8in_verify_ref.npz 2 > /tmp/stageA_bisect.log 2>&1 &
echo started $!
