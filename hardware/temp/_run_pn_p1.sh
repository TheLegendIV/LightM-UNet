cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
ls -l /tmp/golden_pp/case0/p1_in.raw
B=finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/intermediate_models/supported_op_partitions
nohup python3 golden_per_node.py --model $B/partition_1_postfifo_autosize.onnx --input-raw /tmp/golden_pp/case0/p1_in.raw --in-shape 1,128,128,4 --jobs 8 > /tmp/golden_pn_p1.log 2>&1 &
echo started $!
