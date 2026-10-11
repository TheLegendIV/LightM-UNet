#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
D=finn_deployment_outputs/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200_milpfold_8way_20261009_225105
for k in 5 6; do
  python3 node_by_node_check.py --parts-dir $D/intermediate_models/supported_op_partitions --model-suffix "" \
    --build-tmp finn_build_tmp/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200 \
    --partition $k --golden-dir /tmp/golden_pp_bilinear/case0 --out /tmp/nbn2_bil_p$k > /tmp/nbn2_bil_p$k.log 2>&1
  echo EXIT $? >> /tmp/nbn2_bil_p$k.log
done
