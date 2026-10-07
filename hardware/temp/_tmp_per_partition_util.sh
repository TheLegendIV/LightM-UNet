#!/bin/bash
cd /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.runs
for n in 0 1 2 3 4 6 7; do
  rpt=$(find top_GenericPartition_${n}_0_0_synth_1 -iname "*utilization*synth*.rpt" 2>/dev/null | head -1)
  if [ -z "$rpt" ]; then
    echo "partition $n: NO utilization report found"
    continue
  fi
  lut=$(grep -m1 "CLB LUTs" "$rpt" | awk -F'|' '{print $3}' | tr -d ' ')
  ff=$(grep -m1 "CLB Registers" "$rpt" | awk -F'|' '{print $3}' | tr -d ' ')
  echo "partition $n ($rpt): LUT=$lut FF=$ff"
done
