cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh
O=$(ls -d finn_deployment_outputs/S12_dense_256_u4_u8in_int6_p7mvu_milpfold_probe7_*)
B=$O/intermediate_models/supported_op_partitions
P=finn_deployment_outputs/u8in_signbias2_preamble_20261008_174934/intermediate_models
ls $B | head
ls /tmp/chain_mixed_all/case1/ | head -3
python3 rtlsim_taps.py --build-tmp finn_build_tmp/S12_dense_256_u4_u8in_int6_p7mvu --partition 7 --input /tmp/golden_pp_p7mvu/case1/p7_in.raw --out /tmp/taps_p7mvu > /tmp/taps_p7mvu.log 2>&1
echo taps rc=$?
python3 tap_compare.py --taps /tmp/taps_p7mvu --partition 7 --partition-onnx $B/partition_7.onnx --sw-onnx $P/assign_stage_partition_ids_8way.onnx --image-u8 /tmp/chain_mixed_all/case1/in.raw > /tmp/tap_compare_p7mvu.log 2>&1
echo cmp rc=$?
tail -40 /tmp/tap_compare_p7mvu.log | cut -c1-300
