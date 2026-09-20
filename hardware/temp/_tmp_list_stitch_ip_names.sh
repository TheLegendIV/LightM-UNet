#!/bin/bash
for d in /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_stitch_proj_*; do
    xml="$d/ip/component.xml"
    if [ -f "$xml" ]; then
        name=$(grep -m1 '<spirit:name>' "$xml" | sed -n 's#.*<spirit:name>\(.*\)</spirit:name>#\1#p')
        mt=$(stat -c '%y' "$xml")
        echo "$d  ->  $name  ($mt)"
    fi
done
