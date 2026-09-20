#!/bin/bash
echo "=== kernel (3i38u4p5) m_axis_0 busif parameters ==="
xml="/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/zynqbuild_partition7/vivado_stitch_proj_3i38u4p5/ip/component.xml"
grep -B2 -A40 '<spirit:name>m_axis_0</spirit:name>' "$xml" | grep -A2 -i 'TDATA_NUM_BYTES'

echo ""
echo "=== odma (bv2ugwy2) s_axis_0 busif parameters ==="
xml2="/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/zynqbuild_partition7/vivado_stitch_proj_bv2ugwy2/ip/component.xml"
grep -B2 -A40 '<spirit:name>s_axis_0</spirit:name>' "$xml2" | grep -A2 -i 'TDATA_NUM_BYTES'
