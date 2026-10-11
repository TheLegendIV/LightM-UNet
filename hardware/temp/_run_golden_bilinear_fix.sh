#!/bin/bash
# Golden check of the rebuilt P5/P6 using a view dir that hides the pre-fix vivado_stitch_proj dirs.
cd /home/thelegendiv/finn/notebooks/enet
B=finn_build_tmp/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200
V=/tmp/bil_dwfix_view
rm -rf $V; mkdir -p $V
for p in 5 6; do
  mkdir -p $V/GenericPartition_$p
  newest=$(ls -td $PWD/$B/GenericPartition_$p/vivado_stitch_proj_* | head -1)
  for e in $PWD/$B/GenericPartition_$p/*; do
    n=$(basename $e)
    case $n in vivado_stitch_proj_*|rtlsim_golden) continue;; esac
    ln -s $e $V/GenericPartition_$p/$n
  done
  ln -s $newest $V/GenericPartition_$p/$(basename $newest)
  echo "P$p stitch: $newest"; ls -ld --time-style=long-iso $PWD/$B/GenericPartition_$p/rtlsim_single
done
python3 golden_per_partition.py \
  --preamble finn_deployment_outputs/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_preamble_20261009_220505 \
  --build-tmp $V \
  --ref quantEnet_S12_dense_256_u4_bilinear_analytical_v1_ft15ep_u8in_verify_ref.npz \
  --partitions 5-6 --cases 0,1 --out /tmp/golden_pp_bilinear_fix > /tmp/golden_pp_bilinear_fix.log 2>&1
echo EXIT $?
tail -25 /tmp/golden_pp_bilinear_fix.log | cut -c1-220
