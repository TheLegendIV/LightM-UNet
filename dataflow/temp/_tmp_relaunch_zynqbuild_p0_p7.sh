#!/bin/bash
set -e
cd /home/thelegendiv/finn/notebooks/enet
PREAMBLE=finn_deployment_outputs/S12_dense_256_u4_analytical_v1_ft15ep_preamble_20261005_220558
TAG=S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix
CONVORDER=quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_conv_order.json
FOLDING=layer_bits_folding_S12_dense_256_u4_analytical_v1_int6_fps250_lat200_20261006.json
mkdir -p finn_deployment_outputs/zynqbuild_logs
rm -rf finn_build_tmp/zynqbuild_partition0_v2 finn_build_tmp/zynqbuild_partition7_v2

HOME=/tmp/home_dir FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/zynqbuild_partition0_v2 \
  nohup python3 -u finn_zynqbuild_S12_dense_256_u4_analytical_v1_partitionN.py 0 "$PREAMBLE" \
  --tag "$TAG" --conv-order "$CONVORDER" --folding-json "$FOLDING" \
  > finn_deployment_outputs/zynqbuild_logs/partition0_v2.log 2>&1 &
echo "partition0 v2 launched PID $!"

HOME=/tmp/home_dir FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/zynqbuild_partition7_v2 \
  nohup python3 -u finn_zynqbuild_S12_dense_256_u4_analytical_v1_partitionN.py 7 "$PREAMBLE" \
  --tag "$TAG" --conv-order "$CONVORDER" --folding-json "$FOLDING" \
  > finn_deployment_outputs/zynqbuild_logs/partition7_v2.log 2>&1 &
echo "partition7 v2 launched PID $!"

sleep 2
ps aux | grep '[f]inn_zynqbuild_S12_dense_256_u4_analytical_v1_partitionN'
