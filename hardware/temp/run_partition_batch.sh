#!/bin/bash
set -e
cd /home/thelegendiv/finn/notebooks/enet
for i in "$@"; do
  python3 run_single_partition_rtlsim.py "$i" > /tmp/part_${i}_rtlsim.log 2>&1 &
done
wait
echo ALL_DONE
