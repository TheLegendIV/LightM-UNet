#!/bin/bash
# Unattended pipeline: wait for the already-running w16 dummy preamble to
# finish, verify it actually succeeded (final step's intermediate model
# exists), then automatically launch the full 8-way build in the background.
# Run detached inside the FINN container with:
#   docker exec -d -e HOME=/tmp/home_dir <container> bash \
#       /home/thelegendiv/finn/notebooks/enet/finn_w16_dummy_auto_pipeline.sh
set -u
ENET_DIR=/home/thelegendiv/finn/notebooks/enet
PREAMBLE_DIR=/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/hawq_8_2_w16_acc2x_dummy_preamble_20260901_231212
SUCCESS_MARKER="$PREAMBLE_DIR/intermediate_models/assign_stage_partition_ids_8way.onnx"
PIPELINE_LOG=/tmp/hawq_8_2_w16_acc2x_dummy_auto_pipeline.log
FULL_BUILD_LOG=/tmp/hawq_8_2_w16_acc2x_dummy_8way_full.log

{
  echo "=== auto pipeline started $(date) ==="

  PID=$(pgrep -f finn_hawq_preamble_8_2_relu_no_reg_w16_acc2x.py | head -1)
  if [ -n "$PID" ]; then
    echo "waiting for preamble PID $PID to exit..."
    while kill -0 "$PID" 2>/dev/null; do
      sleep 20
    done
  fi
  echo "preamble process exited at $(date)"

  if [ ! -f "$SUCCESS_MARKER" ]; then
    echo "ERROR: success marker $SUCCESS_MARKER not found -- preamble did not complete cleanly. Aborting, NOT launching full build."
    exit 1
  fi
  echo "preamble succeeded (found $SUCCESS_MARKER)"

  echo "=== launching full 8-way build at $(date) ==="
  cd "$ENET_DIR" || exit 1
  nohup python3 finn_ooc_8_2_relu_no_reg_w16_acc2x_8way_full.py "$PREAMBLE_DIR" > "$FULL_BUILD_LOG" 2>&1 &
  BUILD_PID=$!
  echo "full build launched, PID=$BUILD_PID, log=$FULL_BUILD_LOG"

  wait "$BUILD_PID"
  echo "=== full build finished at $(date) with exit code $? ==="
} >> "$PIPELINE_LOG" 2>&1
