#!/bin/bash
for d in ydzflm07 4c00t1f9 equ121xj; do
    xml="/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_stitch_proj_$d/ip/component.xml"
    echo "=== $d ==="
    grep -oE '<spirit:name>[A-Za-z0-9_]+</spirit:name>' "$xml" | sort -u | grep -iE 'axi|dma|m_axis|s_axis'
    echo
done
