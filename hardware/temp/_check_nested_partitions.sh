#!/bin/bash
date
echo ---ALL-GENERICPARTITION-DIRS-ANY-DEPTH---
find /home/thelegendiv/finn/notebooks/enet/finn_build_tmp -maxdepth 4 -type d -name 'GenericPartition_*'
echo ---DONE/FAILED-MARKERS---
grep -n -E 'partition [0-9]+ done|partition [0-9]+ FAILED|Traceback|All partitions' /tmp/finn_fifo_depths_only_512x512.log
