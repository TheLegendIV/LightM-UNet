#!/bin/bash
export HOME=/tmp/home_dir
cd /home/thelegendiv/finn/notebooks/enet
B=finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/intermediate_models/supported_op_partitions
python3 /tmp/check_milp_vs_landed_folding.py /tmp/layer_bits_folding_final.json $B/partition_1_postfifo_autosize.onnx --conv-order /tmp/conv_order.json > /tmp/milp_vs_landed_p1.log 2>&1
echo exit=$?
cut -c1-250 /tmp/milp_vs_landed_p1.log | head -70
