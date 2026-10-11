#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
python3 golden_per_partition.py \
  --preamble finn_deployment_outputs/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_preamble_20261009_220505 \
  --build-tmp finn_build_tmp/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200 \
  --ref quantEnet_S12_dense_256_u4_bilinear_analytical_v1_ft15ep_u8in_verify_ref.npz \
  --cases 0,1 --out /tmp/golden_pp_bilinear > /tmp/golden_pp_bilinear.log 2>&1
echo EXIT $? >> /tmp/golden_pp_bilinear.log
