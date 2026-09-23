#!/bin/bash
for d in /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/GenericPartition_0 \
         /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/GenericPartition_1 \
         /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/GenericPartition_2 \
         /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/GenericPartition_3; do
    n=$(find "$d" -maxdepth 1 -iname 'vivado_stitch_proj*' 2>/dev/null | wc -l)
    m=$(find "$d" -maxdepth 1 -iname 'code_gen_ipgen*' 2>/dev/null | wc -l)
    echo "$d stitch=$n ipgen=$m"
done
grep -c "partition.*done\|FAILED" /tmp/finn_fifo_depths_only_512x512.log
