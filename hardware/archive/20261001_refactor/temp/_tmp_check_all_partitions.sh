#!/bin/bash
OUTDIR=/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_nearest_conv_upsample_256_trained_rtl_mvau_8way_full_v1_256x256_20260926_153428
for i in 0 1 2 3 4 5 6 7; do
  f="$OUTDIR/intermediate_models/supported_op_partitions/partition_$i.onnx"
  echo "partition $i:"
  python3 /tmp/_tmp_check_partition_optypes.py "$f"
done
