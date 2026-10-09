cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
source /tools/Xilinx/Vivado/2022.2/settings64.sh
python3 golden_per_partition.py --preamble finn_deployment_outputs/u8in_signbias2_preamble_20261008_174934 \
  --build-tmp finn_build_tmp/S12_dense_256_u4_u8in_int6_p1fix \
  --ref quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep_u8in_verify_ref.npz \
  --cases 0,1 --partitions 1 --out /tmp/golden_pp_p1fix > /tmp/golden_p1fix.log 2>&1
echo rc=$? >> /tmp/golden_p1fix.log
