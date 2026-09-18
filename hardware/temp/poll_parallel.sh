#!/bin/bash
for p in 3 4 5 6 7; do
  f=/tmp/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_partition${p}_parallel.log
  echo "=== partition $p ==="
  tail -c 300 "$f" 2>/dev/null
  echo
done
