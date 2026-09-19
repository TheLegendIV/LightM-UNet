#!/bin/bash
for d in $(ls -dt /tmp/finn_dev_thelegendiv/vivado_stitch_proj_* 2>/dev/null); do
  echo "== $d =="
  ls -la "$d" 2>/dev/null | grep -E '\.dcp$|\.bit$|\.log$' 
  tail -c 300 "$d/vivado.log" 2>/dev/null
  echo
done
