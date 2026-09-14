#!/bin/bash
set -e
cd /home/thelegendiv/finn/notebooks/enet
nohup python3 finn_build_probe_upsample_nearest_depthwise_int8.py \
    > /tmp/probe_upsample_nearest_depthwise.log 2>&1 &
echo "LAUNCHED_PID=$!"
