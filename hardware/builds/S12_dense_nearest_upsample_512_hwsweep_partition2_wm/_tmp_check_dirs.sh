#!/bin/bash
DIRS="dsrSweep_pbiOff_dsr1.5_preamble_20260929_042814 dsrSweep_pbiOff_dsr1.5_preamble_20260929_042818 dsrSweep_pbiOff_dsr3.0_preamble_20260929_043813 dsrSweep_pbiOff_dsr3.0_preamble_20260929_043814 dsrSweep_pbiOff_dsr5.0_preamble_20260929_044809 dsrSweep_pbiOff_dsr5.0_preamble_20260929_044827 pbiSweep_dsrOff_pbi1.5_preamble_20260929_045802 pbiSweep_dsrOff_pbi1.5_preamble_20260929_045818"
BASE=/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs
for d in $DIRS; do
  f="$BASE/S12_dense_nearest_upsample_512_hwsweep_wm_${d}/hawq_folding_config_partition2.json"
  if [ -f "$f" ]; then
    sz=$(ls -la "$f" | awk '{print $5}')
    echo "$d : OK ${sz} bytes"
  else
    echo "$d : MISSING"
  fi
done
