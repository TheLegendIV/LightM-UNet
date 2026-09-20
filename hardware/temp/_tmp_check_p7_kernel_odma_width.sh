#!/bin/bash
echo "=== kernel (3i38u4p5) m_axis_0 ==="
xml="/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/zynqbuild_partition7/vivado_stitch_proj_3i38u4p5/ip/component.xml"
echo "--- port vector width for m_axis_0_tdata ---"
grep -A20 '<spirit:name>m_axis_0_tdata</spirit:name>' "$xml" | grep -E 'left|right'
echo "--- TDATA_NUM_BYTES busif parameter ---"
grep -B5 -A5 'TDATA_NUM_BYTES' "$xml" | grep -A5 'm_axis_0'

echo ""
echo "=== odma (bv2ugwy2) s_axis_0 ==="
xml2="/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/zynqbuild_partition7/vivado_stitch_proj_bv2ugwy2/ip/component.xml"
echo "--- port vector width for s_axis_0_tdata ---"
grep -A20 '<spirit:name>s_axis_0_tdata</spirit:name>' "$xml2" | grep -E 'left|right'
echo "--- TDATA_NUM_BYTES busif parameter ---"
grep -B5 -A5 'TDATA_NUM_BYTES' "$xml2" | grep -A5 's_axis_0'
