#!/bin/bash
echo "=== finn_build_tmp top level ==="
ls -la --time-style=full-iso /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/
echo "=== per-GenericPartition dir creation/mod times ==="
for d in /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/GenericPartition_*; do
  echo "--- $d ---"
  stat --format='dir mtime: %y' "$d" 2>/dev/null
  find "$d" -maxdepth 2 -iname '*.onnx' -newermt '2026-09-23 20:47:00' 2>/dev/null | head -3
done
