#!/bin/bash
E=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
for b in S12_dense_256_u4_u8in_int6_p7mvu2 S12_dense_256_u4_u8in_int6_fps250_lat200 S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200; do
  echo "== $b P7 stitched port widths"
  f=$(ls -d $E/$b/GenericPartition_7/vivado_stitch_proj_*/ip/component.xml | head -1)
  grep -a -A12 '<spirit:name>m_axis_0_tdata</spirit:name>' $f | grep -a -E 'left|right' | head -2
  grep -a -A12 '<spirit:name>s_axis_0_tdata</spirit:name>' $f | grep -a -E 'left|right' | head -2
done
echo "== ILA slot 7 width: original vs new project"
for P in /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical /home/thelegendiv/finn/vivado_projects/S12_256_analytical_p7mvu2/s12_256_analytical_p7mvu2; do
  echo "-- $P"
  grep -a -n 'SLOT_7_AXIS_TDATA_WIDTH\|C_SLOT_7_AXIS_TDATA_WIDTH\|SLOT_7_AXIS_tdata' $P/*.srcs/sources_1/bd/top/top.bd | head -5
done
echo "== 8-bit warning in original project logs?"
grep -a -l 'SLOT_7_AXIS_tdata' /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/*.log /tmp/*.log 2>/dev/null | head
