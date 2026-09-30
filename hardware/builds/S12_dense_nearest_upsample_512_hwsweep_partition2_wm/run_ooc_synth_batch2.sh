#!/bin/bash
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet

run_one() {
  TAG="$1"
  FN="$2"
  OUTDIR=$(dirname "$FN")
  LOG="/tmp/oocsynth_${TAG}.log"
  nohup python3 finn_ooc_synth_only.py "$FN" "$OUTDIR" "$TAG" > "$LOG" 2>&1 &
  echo "LAUNCHED $TAG PID=$!"
}

run_one dsr5.0 "finn_deployment_outputs/S12_dense_nearest_upsample_512_hwsweep_wm_dsrSweep_pbiOff_dsr5.0_milpfold_partition2_20260929_172638/partition2_dsrSweep_pbiOff_dsr5.0_milpfold_stitched.onnx"
run_one pbi1.5 "finn_deployment_outputs/S12_dense_nearest_upsample_512_hwsweep_wm_pbiSweep_dsrOff_pbi1.5_milpfold_partition2_20260929_172638/partition2_pbiSweep_dsrOff_pbi1.5_milpfold_stitched.onnx"
