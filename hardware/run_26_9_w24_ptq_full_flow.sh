#!/bin/bash
# Orchestrates the full unattended 26_9_w24_s14w12_nonneg_block PTQ deployment
# pipeline inside the FINN container: preamble (tidy/streamline/convert_to_hw/
# 8-way partition tagging) -> full 8-way folding-bridge + DSP-forced build ->
# combine -> OOC synthesis. Single blocking script so it can be launched once
# via nohup and left running unattended.
set -e
cd /home/thelegendiv/finn/notebooks/enet

echo "=== [1/2] Running preamble ===" 
PREAMBLE_LOG=/tmp/hawq_26_9_w24_ptq_preamble.log
python3 finn_hawq_preamble_26_9_w24_ptq_joint.py > "$PREAMBLE_LOG" 2>&1
PREAMBLE_DIR=$(grep -o 'OUTPUT_DIR= .*' "$PREAMBLE_LOG" | tail -1 | cut -d' ' -f2)
if [ -z "$PREAMBLE_DIR" ]; then
    echo "FATAL: could not find OUTPUT_DIR in $PREAMBLE_LOG"
    exit 1
fi
echo "Preamble done. PREAMBLE_DIR=$PREAMBLE_DIR"

echo "=== [2/2] Running full 8-way build (bridge + DSP-force + build + combine + OOC synth) ==="
BUILD_LOG=/tmp/hawq_26_9_w24_ptq_8way_full.log
python3 finn_ooc_26_9_w24_ptq_joint_8way_full.py "$PREAMBLE_DIR" > "$BUILD_LOG" 2>&1
echo "Full 8-way build done. See $BUILD_LOG"
