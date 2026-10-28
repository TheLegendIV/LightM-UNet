#!/bin/bash
for f in partition5_refix2 partition6_v2 preamble_v2; do
  echo "== $f"
  grep -E 'DEEP FIFO|DEEP FORCED|RESULT|FAILED|Traceback|Error:|dangling-node' /tmp/$f.log | cut -c1-500
  tail -n 1 /tmp/$f.log | cut -c1-200
done
pgrep -af "rebuild_partition|finn_s12_preamble" | grep -v pgrep | cut -c1-150
