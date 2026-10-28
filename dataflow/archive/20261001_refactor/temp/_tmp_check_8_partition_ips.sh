#!/bin/bash
BASE=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
for d in vivado_stitch_proj_6_75nd4k vivado_stitch_proj_4ti661rj vivado_stitch_proj_e_z5561z \
         vivado_stitch_proj_9qynp0ng vivado_stitch_proj_ohvum343 vivado_stitch_proj_anxxap8z \
         vivado_stitch_proj_5kxs5a6b vivado_stitch_proj_btoactkv; do
    if [ -f "$BASE/$d/ip/component.xml" ]; then
        echo "OK: $d"
    else
        echo "MISSING: $d"
    fi
done
