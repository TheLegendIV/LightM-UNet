#!/bin/bash
REPORT_DIR=/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_20260918_144349/report
mkdir -p "$REPORT_DIR"
echo "pid=2186957 partitions=[4, 5, 6, 7] (manually created for already-running early job launched before marker-file logic existed)" > "$REPORT_DIR/.early_ooc_synth_in_progress"
cat "$REPORT_DIR/.early_ooc_synth_in_progress"
