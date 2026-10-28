#!/bin/bash
BASE=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix
echo "--- partition 0 IODMA ---"
find "$BASE/GenericPartition_0" -iname '*IODMA*' -maxdepth 2
echo "--- partition 7 IODMA ---"
find "$BASE/GenericPartition_7" -iname '*IODMA*' -maxdepth 2
