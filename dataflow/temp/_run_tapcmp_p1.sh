cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
B=finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/intermediate_models/supported_op_partitions
P=finn_deployment_outputs/u8in_signbias2_preamble_20261008_174934/intermediate_models
python3 tap_compare.py --taps /tmp/taps_p1 --partition 1 --partition-onnx $B/partition_1.onnx --sw-onnx $P/assign_stage_partition_ids_8way.onnx --image-u8 /tmp/golden_in/test_1_p0000_0000_input_u8.raw > /tmp/tap_compare_p1.log 2>&1
echo rc=$?
