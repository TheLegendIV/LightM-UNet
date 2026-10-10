#!/bin/bash
# waits up to $1 seconds for verify_export.py to exit
for i in $(seq 1 $(( ${1:-600} / 10 ))); do
  ps -eo args | grep 'python3 verify_export' | grep -v grep > /dev/null || { echo "verify exited"; date +%H:%M:%S; exit 0; }
  sleep 10
done
echo "still running"; date +%H:%M:%S
