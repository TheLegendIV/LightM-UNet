#!/bin/bash
for p in 0 1 2 3 4 5 6 7; do
  echo -n "partition $p: "
  grep vivado_proj_folder /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v2_20260918/report/ooc_synth_partition_${p}.json
done
