#!/bin/bash
D=/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200_milpfold_8way_20261009_225105
B=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200
echo "== deploy dir"; ls $D; ls $D/report | head -20
echo "== build tmp"; ls $B
for p in 0 1 2 3 4 5 6 7; do
  echo "-- P$p"; ls -d $B/GenericPartition_$p/vivado_stitch_proj_* $B/GenericPartition_$p/code_gen_ipgen_IODMA* 2>/dev/null
done
echo "== iodma/dma sub-dirs anywhere in deploy dir"; find $D -maxdepth 2 -iname '*iodma*' | head
echo "== stitched ip dirs have component.xml?"
ls $B/GenericPartition_0/vivado_stitch_proj_*/ip/component.xml
echo "== existing projects referencing bilinear build"
grep -c "S12_dense_256_u4_bilinear" /home/thelegendiv/finn/vivado_projects/*/*/*.xpr
echo "== existing 256 proj top.bd IODMA refs"
ls /home/thelegendiv/finn/vivado_projects/
