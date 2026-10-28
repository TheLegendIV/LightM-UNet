#!/bin/bash
for i in 1 2 3 4; do
  echo "==p${i}=="
  tail -5 "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/regen_stitched_ip_partition${i}.log"
done
