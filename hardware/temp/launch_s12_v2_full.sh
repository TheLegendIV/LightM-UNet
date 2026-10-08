#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
nohup python3 -u finn_s12_build.py /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_analytical_v2_ft15ep_preamble_20261008_032400 \
  --tag S12_dense_256_u4_analytical_v2_ft15ep_int6_fps250_lat200 \
  --conv-order quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_conv_order.json \
  --folding-json layer_bits_folding_S12_dense_256_u4_analytical_v2_int6_fps250_lat200.json \
  --partitions all > /tmp/ooc_S12_v2_full.log 2>&1 &
echo launched $!
