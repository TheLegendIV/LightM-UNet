#!/bin/bash
E=/home/thelegendiv/finn/notebooks/enet
echo "== deployment outputs (S12 256 analytical, newest first)"
ls -dt $E/finn_deployment_outputs/*S12_dense_256_u4_analytical* 2>/dev/null | head -30
echo "== vivado_projects"
ls -la /home/thelegendiv/finn/vivado_projects/ /home/thelegendiv/finn/vivado_projects/S12_256_analytical $E/vivado_projects 2>&1 | head -60
echo "== finn_build_tmp"
ls -dt $E/finn_build_tmp/* | head -30
echo "== partition 7 dirs"
ls -dt $E/finn_build_tmp/*/GenericPartition_7 2>/dev/null
