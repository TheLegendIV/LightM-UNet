#!/bin/bash
declare -A PROJ=(
  [0]="synth_out_of_context_e6zkir5w"
  [1]="synth_out_of_context_wyssjzo6"
  [2]="synth_out_of_context_m3tho2sv"
  [3]="synth_out_of_context_r6cyhxhn"
  [4]="synth_out_of_context_l5h4hg2r"
  [5]="synth_out_of_context_2fojftrv"
  [6]="synth_out_of_context_sr4sdn9l"
  [7]="synth_out_of_context_gte8kz3p"
)
for i in 0 1 2 3 4 5 6 7; do
  proj=${PROJ[$i]}
  rpt="/tmp/finn_dev_thelegendiv/${proj}/results_GenericPartition_${i}_wrapper/vivadocompile/vivadocompile.runs/impl_1/GenericPartition_${i}_wrapper_utilization_placed.rpt"
  echo "=== partition $i ($rpt) ==="
  if [ -f "$rpt" ]; then
    grep -A 6 "^1. Slice Logic Distribution\|^3. DSP$\|^3\. DSP\b" "$rpt" | head -20
    echo "--- DSP48E2 line ---"
    grep -i "DSP48E2\|DSPs" "$rpt"
  else
    echo "MISSING"
  fi
  echo
done
