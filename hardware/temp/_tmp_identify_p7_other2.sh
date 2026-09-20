#!/bin/bash
for d in 3i38u4p5 bv2ugwy2; do
    xml="/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/zynqbuild_partition7/vivado_stitch_proj_${d}/ip/component.xml"
    echo "=== $d ==="
    grep -oE '<spirit:name>[A-Za-z0-9_]+</spirit:name>' "$xml" | sort -u | grep -iE '^<spirit:name>(m_axi_gmem0|m_axis_0|s_axis_0|s_axi_control_0)</spirit:name>$'
done
