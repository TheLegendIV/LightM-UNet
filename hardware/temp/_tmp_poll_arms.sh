#!/bin/bash
echo "=== run_arms_full.log tail ==="
tail -n 30 /tmp/run_arms_full.log
echo "=== per-arm tails ==="
for t in ratchet_ablation_finn_autofold ratchet_1pct ratchet_25pct ratchet_100pct ratchet_200pct ratchet_off analytical_25pct analytical_25pct_finnfifo; do
  echo "--- $t ---"
  tail -n 6 /tmp/ooc_${t}.log 2>&1
done
