#!/bin/bash
# Batch 1 of 2 (4 of 6 tags) -- resume OOC synth from the already-computed
# prefifo_autosize checkpoints (FIFO depths already valid, skips the
# expensive rtlsim autosizing step). See finn_ooc_resume_from_prefifo_partition2.py.
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet

run_one() {
  CKPT="$1"
  OUTDIR=$(dirname "$CKPT")
  LOG="$OUTDIR/resume_ooc.log"
  nohup python3 finn_ooc_resume_from_prefifo_partition2.py "$CKPT" > "$LOG" 2>&1 &
  echo "LAUNCHED $CKPT PID=$! (log: $LOG)"
}

run_one "finn_deployment_outputs/S12_dense_nearest_upsample_512_hwsweep_wm_baseline_both_off_autofold_partition2_20260929_172638/partition2_baseline_both_off_autofold_prefifo_autosize.onnx"
run_one "finn_deployment_outputs/S12_dense_nearest_upsample_512_hwsweep_wm_baseline_both_off_milpfold_partition2_20260929_172638/partition2_baseline_both_off_milpfold_prefifo_autosize.onnx"
run_one "finn_deployment_outputs/S12_dense_nearest_upsample_512_hwsweep_wm_dsrSweep_pbiOff_dsr1.5_milpfold_partition2_20260929_172638/partition2_dsrSweep_pbiOff_dsr1.5_milpfold_prefifo_autosize.onnx"
run_one "finn_deployment_outputs/S12_dense_nearest_upsample_512_hwsweep_wm_dsrSweep_pbiOff_dsr3.0_milpfold_partition2_20260929_172638/partition2_dsrSweep_pbiOff_dsr3.0_milpfold_prefifo_autosize.onnx"

disown -a
echo "batch1 (4 tags) launched"
