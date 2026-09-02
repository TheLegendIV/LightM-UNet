#!/bin/bash
# Chains the no-Vivado preamble -> full 8-way build+OOC-synth for the DSC-no-
# projection/dense_dilation 8_2_relu_no_reg_w20 HAWQ acc2x dummy (random-
# weight) deployment, so both stages run unattended in one background job.
set -e
cd /home/thelegendiv/finn/notebooks/enet

PREAMBLE_LOG=/tmp/hawq_8_2_w20_acc2x_dummy_preamble.log
FULL_LOG=/tmp/hawq_8_2_w20_acc2x_dummy_8way_full.log
CHAIN_LOG=/tmp/hawq_8_2_w20_acc2x_dummy_chain.log

echo "[chain] $(date) starting preamble" >> "$CHAIN_LOG"
python3 finn_hawq_preamble_8_2_relu_no_reg_w20_acc2x.py > "$PREAMBLE_LOG" 2>&1

PREAMBLE_DIR=$(grep -oP '(?<=^OUTPUT_DIR= ).*' "$PREAMBLE_LOG" | tail -1)
echo "[chain] $(date) preamble done, dir=$PREAMBLE_DIR" >> "$CHAIN_LOG"
if [ -z "$PREAMBLE_DIR" ]; then
    echo "[chain] $(date) ERROR: could not parse preamble OUTPUT_DIR, aborting" >> "$CHAIN_LOG"
    exit 1
fi

echo "[chain] $(date) starting full 8-way build + OOC synth" >> "$CHAIN_LOG"
python3 finn_ooc_8_2_relu_no_reg_w20_acc2x_8way_full.py "$PREAMBLE_DIR" > "$FULL_LOG" 2>&1
echo "[chain] $(date) full 8-way build finished, exit=$?" >> "$CHAIN_LOG"
