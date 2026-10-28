#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs
for d in ratchet_ablation_finn_autofold_autofold_partition2_20261009_154752 ratchet_1pct_simfifo_milpfold_partition2_20261009_154757 ratchet_25pct_simfifo_milpfold_partition2_20261009_154802 ratchet_100pct_simfifo_milpfold_partition2_20261009_155407 ratchet_200pct_simfifo_milpfold_partition2_20261009_175218 ratchet_off_simfifo_milpfold_partition2_20261009_180323; do
  echo "## $d"
  tr -d '\n ' < $d/report/ooc_synth_and_timing.json | cut -c1-500; echo
  ls $d/report | tr '\n' ' '; echo
  f=$(find $d -name vivado.log 2>/dev/null | head -1); echo "vivado.log: $f"
  [ -n "$f" ] && grep -E "^\| *(DSPs|DSP48 Blocks|CLB LUTs|Block RAM Tile) " $f | tail -4
done
