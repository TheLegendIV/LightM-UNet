#!/bin/bash
BASE=/tmp/finn_dev_thelegendiv
for d in "$BASE"/synth_out_of_context_*; do
  # find the results_GenericPartition_X_wrapper subdir
  rd=$(find "$d" -maxdepth 1 -iname "results_GenericPartition_*" 2>/dev/null | head -n1)
  [ -z "$rd" ] && continue
  wrapper=$(basename "$rd")
  restxt="$rd/res.txt"
  placed_rpt=$(find "$rd" -iname "*_utilization_placed.rpt" 2>/dev/null | head -n1)
  synth_rpt=$(find "$rd" -iname "*_utilization_synth.rpt" 2>/dev/null | head -n1)
  if [ ! -f "$restxt" ]; then
    echo "$d -> $wrapper : NO res.txt (crashed run), skipping"
    continue
  fi
  dsp_line=""
  if [ -n "$placed_rpt" ]; then
    dsp_line=$(grep "DSP48E2" "$placed_rpt" | head -n1)
  fi
  if [ -z "$dsp_line" ] && [ -n "$synth_rpt" ]; then
    dsp_line=$(grep "DSP48E2" "$synth_rpt" | head -n1)
  fi
  echo "$d -> $wrapper : real_dsp_line=[$dsp_line]"
done
