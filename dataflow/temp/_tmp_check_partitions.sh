#!/bin/bash
BD=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix
for i in 0 1 2 3 4 5 6 7; do
  if [ -d "$BD/GenericPartition_$i" ]; then
    echo "GenericPartition_$i: EXISTS"
  else
    echo "GenericPartition_$i: MISSING"
  fi
  if [ -e "$BD/GenericPartition_${i}_r_GenericPartition_${i}" ]; then
    echo "  mangled_${i}: ALREADY EXISTS (symlink or dir)"
  else
    echo "  mangled_${i}: absent (safe to symlink)"
  fi
done
