#!/bin/bash
# Batch 2 of 2 (remaining 2 of 6 tags) -- run after batch1 finishes.
# See finn_ooc_resume_from_prefifo_partition2.py.
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet

run_one() {
  CKPT="$1"
  OUTDIR=$(dirname "$CKPT")
  LOG="$OUTDIR/resume_ooc.log"
  nohup python3 finn_ooc_resume_from_prefifo_partition2.py "$CKPT" > "$LOG" 2>&1 &
  echo "LAUNCHED $CKPT PID=$! (log: $LOG)"
}

run_one "finn_deployment_outputs/S12_dense_nearest_upsample_512_hwsweep_wm_dsrSweep_pbiOff_dsr5.0_milpfold_partition2_20260929_172638/partition2_dsrSweep_pbiOff_dsr5.0_milpfold_prefifo_autosize.onnx"
run_one "finn_deployment_outputs/S12_dense_nearest_upsample_512_hwsweep_wm_pbiSweep_dsrOff_pbi1.5_milpfold_partition2_20260929_172638/partition2_pbiSweep_dsrOff_pbi1.5_milpfold_prefifo_autosize.onnx"

disown -a
echo "batch2 (2 tags) launched"
