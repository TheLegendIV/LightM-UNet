#!/bin/bash
for d in /tmp/finn_dev_thelegendiv/code_gen_ipgen_*_hls_*; do
  if [ -d "$d" ]; then
    n=$(find "$d" -iname component.xml | wc -l)
    if [ "$n" -eq 0 ]; then
      echo "BROKEN (no component.xml): $d"
    fi
  fi
done
echo "--- done ---"
