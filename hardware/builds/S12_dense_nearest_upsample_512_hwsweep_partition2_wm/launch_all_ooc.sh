#!/bin/bash
# Launch all 6 real Vivado OOC-synthesis builds in parallel (background, nohup).
set -e
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet

ENET_DIR=/home/thelegendiv/finn/notebooks/enet
OUT=$ENET_DIR/finn_deployment_outputs

declare -A PREAMBLE_DIRS=(
  [baseline_both_off]="$OUT/S12_dense_nearest_upsample_512_hwsweep_wm_baseline_both_off_preamble_20260929_041647"
  [dsrSweep_pbiOff_dsr1.5]="$OUT/S12_dense_nearest_upsample_512_hwsweep_wm_dsrSweep_pbiOff_dsr1.5_preamble_20260929_042814"
  [dsrSweep_pbiOff_dsr3.0]="$OUT/S12_dense_nearest_upsample_512_hwsweep_wm_dsrSweep_pbiOff_dsr3.0_preamble_20260929_043813"
  [dsrSweep_pbiOff_dsr5.0]="$OUT/S12_dense_nearest_upsample_512_hwsweep_wm_dsrSweep_pbiOff_dsr5.0_preamble_20260929_044809"
  [pbiSweep_dsrOff_pbi1.5]="$OUT/S12_dense_nearest_upsample_512_hwsweep_wm_pbiSweep_dsrOff_pbi1.5_preamble_20260929_045818"
)

launch() {
  local tag="$1"
  local pdir="$2"
  local fold="$3"   # "milp" or "auto"
  local logtag="$4" # unique log suffix
  local extra_arg=""
  if [ "$fold" = "milp" ]; then
    extra_arg="$pdir/hawq_folding_config_partition2.json"
  fi
  nohup python3 finn_ooc_partition2_trained.py "$pdir" "$tag" $extra_arg \
    > "/tmp/ooc_${logtag}.log" 2>&1 &
  echo "launched $logtag (tag=$tag fold=$fold) pid=$!"
}

for tag in "${!PREAMBLE_DIRS[@]}"; do
  launch "$tag" "${PREAMBLE_DIRS[$tag]}" "milp" "milpfold_${tag}"
done

# extra auto-fold build reusing baseline_both_off's preamble
launch "baseline_both_off" "${PREAMBLE_DIRS[baseline_both_off]}" "auto" "autofold_baseline_both_off"

disown -a
echo "all 6 builds launched (detached via nohup+disown)"
