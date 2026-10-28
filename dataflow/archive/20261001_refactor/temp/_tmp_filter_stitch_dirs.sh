#!/bin/bash
ls -lt --time-style=full-iso /tmp/finn_dev_thelegendiv/ | grep vivado_stitch_proj | while read -r line; do
  d=$(echo "$line" | awk '{print $6}')
  if [[ "$d" > "2026-09-18" || "$d" == "2026-09-18" ]]; then
    echo "$line"
  fi
done
