#!/bin/bash
export HOME=/tmp/home_dir
cd /home/thelegendiv/finn/notebooks/enet
nohup python3 node_by_node_check.py \
  --parts-dir finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/intermediate_models/supported_op_partitions \
  --build-tmp finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200 \
  --partition 1 --golden-dir /tmp/golden_pp/case0 --out /tmp/nbn_p1 > /tmp/nbn_p1_full.log 2>&1 &
echo started $!
