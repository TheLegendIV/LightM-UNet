#!/bin/bash
xml="/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/zynqbuild_partition7/vivado_stitch_proj_5gj9ep0w/ip/component.xml"
grep -oE '<spirit:name>[A-Za-z0-9_]+</spirit:name>' "$xml" | sort -u | grep -iE 'axi|dma|m_axis|s_axis'
