cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
source /tools/Xilinx/Vivado/2022.2/settings64.sh
B=finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/intermediate_models/supported_op_partitions
P=finn_deployment_outputs/u8in_signbias2_preamble_20261008_174934/intermediate_models
nohup bash -c "python3 rtlsim_taps.py --build-tmp finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200 --partition 7 --input /tmp/golden_pp/case0/p7_in.raw --out /tmp/taps_p7 > /tmp/taps_p7.log 2>&1; python3 tap_compare.py --taps /tmp/taps_p7 --partition 7 --partition-onnx $B/partition_7.onnx --sw-onnx $P/assign_stage_partition_ids_8way.onnx --image-u8 /tmp/chain_mixed_all/case0/in.raw > /tmp/tap_compare_p7.log 2>&1; echo done >> /tmp/tap_compare_p7.log" > /dev/null 2>&1 &
