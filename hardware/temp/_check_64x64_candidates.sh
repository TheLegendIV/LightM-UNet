#!/bin/bash
for d in \
  12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_20260923_005951 \
  12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_20260923_010141 \
  12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_20260923_011534 \
  12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_20260923_101905 \
  12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_20260918_140017 \
  12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_20260918_144349 \
  12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_20260917_011115 \
  ; do
  base="/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/$d/intermediate_models/supported_op_partitions"
  n=$(ls "$base" 2>/dev/null | grep -c fifo_sized)
  echo "$d : $n fifo_sized"
done
