cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh
python3 golden_per_partition.py --preamble finn_deployment_outputs/u8in_signbias2_preamble_20261008_174934 \
  --build-tmp finn_build_tmp/S12_dense_256_u4_u8in_int6_p7mvu \
  --ref quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep_u8in_verify_ref.npz \
  --cases 0,1,2,3 --partitions 7 --out /tmp/golden_pp_p7mvu --jobs 2 > /tmp/golden_p7mvu.log 2>&1
echo rc=$? >> /tmp/golden_p7mvu.log
tail -40 /tmp/golden_p7mvu.log
