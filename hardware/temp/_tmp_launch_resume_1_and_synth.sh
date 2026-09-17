#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet && \
nohup bash -c "python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_resume_1.py && \
  python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_partition_synth.py \
  /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_20260917_011115" \
  > /tmp/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_resume_1_and_synth.log 2>&1 &
disown
