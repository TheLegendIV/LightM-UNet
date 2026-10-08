#!/bin/bash
for d in /home/thelegendiv/finn /home/thelegendiv/finn/deps/finn-hlslib /home/thelegendiv/finn/deps/qonnx /home/thelegendiv/finn/deps/brevitas /tmp/home_dir/.local/src/finn /tmp/home_dir/.local/src/qonnx /tmp/home_dir/.local/src/brevitas; do
  if [ -d "$d" ]; then
    echo "== $d"
    git -C "$d" config --global --add safe.directory "$d" 2>/dev/null
    git -C "$d" log -1 --format='%H %ad %s' --date=short 2>&1 | head -2
    git -C "$d" describe --tags --always --dirty 2>&1 | head -1
    git -C "$d" status --short 2>&1 | head -5
  fi
done
