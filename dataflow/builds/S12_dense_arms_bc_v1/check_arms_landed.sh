#!/bin/bash
# Landed-folding gate for the C arms (no Vivado): bridged config -> landed partition-2 graph
# -> check_milp_vs_landed_folding.py vs the MILP json. Exits 1 on any mismatch.
# Usage: bash check_arms_landed.sh [preamble_dir]   (default: waits for the newest lex_dsr2_fps305 preamble's bridge output)
set -uo pipefail
cd /home/thelegendiv/finn/notebooks/enet
PDIR=${1:-$(ls -d finn_deployment_outputs/*_lex_dsr2_fps305_preamble_* | tail -1)}
echo "preamble dir: $PDIR"
rc=0
for arm in dsr2 nodsr; do
  cfg="$PDIR/hawq_folding_config_armC_${arm}.json"
  while [ ! -s "$cfg" ]; do sleep 30; done
  sleep 5  # let the bridge finish writing
  landed="$PDIR/landed_partition2_armC_${arm}.onnx"
  python3 dump_milpfold_landed_partition2.py "$PDIR" "$cfg" "$landed" > "/tmp/arms_landed_${arm}.log" 2>&1 || { echo "armC_${arm}: landed dump FAILED (see /tmp/arms_landed_${arm}.log)"; rc=1; continue; }
  echo "=== armC_${arm} ==="
  python3 check_milp_vs_landed_folding.py "layer_bits_folding_lex_${arm}_fps305.json" "$landed" || rc=1
done
echo "LANDED CHECK: $([ $rc -eq 0 ] && echo OK || echo MISMATCH)"
exit $rc
