#!/bin/bash
export HOME=/tmp/home_dir
B=/home/thelegendiv/finn/notebooks/enet
JOB=S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200
OUTD=$B/finn_deployment_outputs/${JOB}_milpfold_8way_20261009_225105
cd /tmp/calib
python3 build_node_resource_calibration_csv.py \
  --build-dir $B/finn_build_tmp/$JOB \
  --onnx-dir $OUTD/intermediate_models/supported_op_partitions \
  --onnx-pattern 'partition_{i}.onnx' \
  --rpt-cache-dir $OUTD/hier_cache_calib \
  --out $OUTD/node_resource_calibration_S12_dense_256_u4_bilinear_analytical_v1_dwfix.csv
echo "exit=$?"
