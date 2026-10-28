#!/bin/bash
d=/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_20260918_144349
echo "--- top level ---"
ls "$d"
echo "--- intermediate_models ---"
ls "$d/intermediate_models" 2>/dev/null
echo "--- supported_op_partitions ---"
ls "$d/intermediate_models/supported_op_partitions" 2>/dev/null
