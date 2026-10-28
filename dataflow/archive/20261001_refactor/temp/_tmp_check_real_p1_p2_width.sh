#!/bin/bash
echo "=== real Partition_1 (4ti661rj) m_axis_0 ==="
xml="/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_stitch_proj_4ti661rj/ip/component.xml"
echo "-- physical port width --"
grep -A20 '<spirit:name>m_axis_0_tdata</spirit:name>' "$xml" | grep -E 'left|right'
echo "-- TDATA_NUM_BYTES --"
grep -B2 -A40 '<spirit:name>m_axis_0</spirit:name>' "$xml" | grep -A2 -i 'TDATA_NUM_BYTES'

echo ""
echo "=== real Partition_2 (e_z5561z) s_axis_0 ==="
xml2="/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_stitch_proj_e_z5561z/ip/component.xml"
echo "-- physical port width --"
grep -A20 '<spirit:name>s_axis_0_tdata</spirit:name>' "$xml2" | grep -E 'left|right'
echo "-- TDATA_NUM_BYTES --"
grep -B2 -A40 '<spirit:name>s_axis_0</spirit:name>' "$xml2" | grep -A2 -i 'TDATA_NUM_BYTES'
