#!/bin/bash
# Per-node hardware-resource calibration dump of the uniform-INT8 bilinear 512 v2 build: Vivado `report_utilization -hierarchical` on each partition's routed OOC checkpoint joined with the landed
# ONNX node attributes (MVAU/VVAU, Thresholding, SWU, FIFO, DWC, FMPadding, stream nodes) -> ONE csv, row per node.
# Run inside finn_persistent:   docker exec finn_persistent bash /home/thelegendiv/finn/notebooks/enet/_launch_calibration_dump_int8_v2.sh
# Output: $OUTD/node_resource_calibration_S12_dense_512_bilinear_int8_v2.csv (copy to hardware/builds/S12_dense_512_bilinear_int8_v2/results/). Hier reports are cached in $OUTD/hier_cache_calib (re-run reuses them).
export HOME=/tmp/home_dir
B=/home/thelegendiv/finn/notebooks/enet
JOB=S12_dense_512_bilinear_int8_v2
OUTD=$B/finn_deployment_outputs/${JOB}_milpfold_8way_20261016_234515
cd $B/calib_int8_v2
nohup python3 -u build_node_resource_calibration_csv.py \
  --build-dir $B/finn_build_tmp/$JOB \
  --onnx-dir $OUTD/intermediate_models/supported_op_partitions \
  --onnx-pattern 'partition_{i}.onnx' \
  --rpt-cache-dir $OUTD/hier_cache_calib \
  --out $OUTD/node_resource_calibration_${JOB}.csv > $B/calib_int8_v2/dump.log 2>&1 &
echo "launched $!"
