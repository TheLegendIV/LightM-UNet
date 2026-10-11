#!/bin/bash
# Rebuild partitions 5 and 6 of the bilinear 256 build in place (depthwise SWG/VVAU folding fix); 0-4,7 are kept.
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
NAME=S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200_milpfold_8way_20261009_225105
OUTROOT=finn_deployment_outputs
BK=$OUTROOT/archive_pre_20261010_dwfold_fix_$NAME
if [ ! -d "$BK" ]; then cp -a $OUTROOT/$NAME $BK; fi
PDIR=$OUTROOT/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_preamble_20261009_220505
python3 -u finn_s12_build.py $PDIR \
  --tag S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200 \
  --conv-order quantEnet_S12_dense_256_u4_bilinear_analytical_v1_conv_order.json \
  --folding-json layer_bits_folding_S12_dense_256_u4_bilinear_analytical_v1_int6_fps250_lat200.json \
  --partitions 5 6 --probe --output-dir $OUTROOT/$NAME > /tmp/ooc_S12_bilinear_p56_dwfix.log 2>&1
echo "build exit=$?" >> /tmp/ooc_S12_bilinear_p56_dwfix.log
# the probe re-cut every partition file; put the already-built 0-4,7 back
for i in 0 1 2 3 4 7; do
  cp -a $BK/intermediate_models/supported_op_partitions/partition_$i.onnx $OUTROOT/$NAME/intermediate_models/supported_op_partitions/partition_$i.onnx
done
echo "restored 0-4,7" >> /tmp/ooc_S12_bilinear_p56_dwfix.log
