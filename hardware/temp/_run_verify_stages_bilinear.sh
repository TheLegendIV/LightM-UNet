#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
P=finn_deployment_outputs/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_preamble_20261009_220505/intermediate_models
python3 verify_export.py --ref quantEnet_S12_dense_256_u4_bilinear_analytical_v1_ft15ep_u8in_verify_ref.npz --n 2 \
  --onnx streamline=$P/step_enet_streamline.onnx \
         fuse_leaky=$P/step_fuse_leaky_relu_to_threshold.onnx \
         dedup=$P/step_dedup_forked_matmul_before_threshold.onnx \
         compose_thr=$P/step_compose_consecutive_thresholds.onnx \
         hw_rtl_mvau=$P/step_enet_convert_to_hw_rtl_mvau.onnx \
         assign8=$P/assign_stage_partition_ids_8way.onnx > /tmp/verify_bil_stages.log 2>&1
echo EXIT $? >> /tmp/verify_bil_stages.log
