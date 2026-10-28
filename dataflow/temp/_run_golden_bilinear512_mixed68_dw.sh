#!/bin/bash
export HOME=/tmp/home_dir
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
OUT=golden_pp_bilinear512_mixed68_dw_le
nohup python3 -u golden_per_partition.py \
  --preamble finn_deployment_outputs/S12_dense_512_bilinear_mixed68_v1_ft15ep_dw_preamble_20261013_000943 \
  --build-tmp finn_build_tmp/S12_dense_512_bilinear_mixed68_v1_ft15ep_dw \
  --ref quantEnet_S12_dense_512_bilinear_mixed68_v1_ft15ep_u8in_verify_ref.npz \
  --cases 0,1 --out $OUT > $OUT.log 2>&1 &
echo "launched $!"
disown -a
