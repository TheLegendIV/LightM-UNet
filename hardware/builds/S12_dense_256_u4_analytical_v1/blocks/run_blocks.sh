#!/bin/bash
# Per-SHAPE standalone builds of the S12 256x256 U4 net (analytical folding, NOT re-solved): one partition per distinct block shape, each stitched and rtlsim'd on its own.
#   shapes come from block_shapes.json (make_block_shapes.py: 29 blocks -> 12 shapes; one representative block per shape is built).
# Run INSIDE the FINN container (HOME=/tmp/home_dir) after docker cp-ing the inputs into the flat /home/thelegendiv/finn/notebooks/enet/ dir (README.md "Inputs").
# Usage:  bash run_blocks.sh                  all 12 shapes
#         bash run_blocks.sh stage3.7 up4     only these blocks (block names or partition ids)
#         STEP=bridge bash run_blocks.sh      preamble + per-block partitioning + bridge dry run only (no HLS / Vivado / rtlsim): do this first
#         NO_OOC=0 bash run_blocks.sh         also run Vivado out-of-context synthesis per block (default: skipped, rtlsim only)
set -eo pipefail
cd /home/thelegendiv/finn/notebooks/enet
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1

MODEL=quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep
CONV_ORDER=quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_conv_order.json
FOLDING=layer_bits_folding_u4_analytical.json          # MILP/artifacts/S12_dense_256_u4_analytical_v1/int6_fps250_lat200/layer_bits_folding_final.json, copied under this name
SHAPES=block_shapes.json
TAG=blocks256
TARGET_FPS=250
WWIDTH=72
STEP=${STEP:-all}
NO_OOC=${NO_OOC:-1}
PARTS=("$@")
if [ ${#PARTS[@]} -eq 0 ]; then PARTS=(shapes); fi

echo "=== preamble (per-block partitioning) ==="
python3 finn_s12_preamble.py "$MODEL" --tag "$TAG" --blocks --conv-order "$CONV_ORDER" 2>&1 | tee /tmp/${TAG}_preamble.log
PDIR=$(ls -d finn_deployment_outputs/${TAG}_preamble_* | tail -1)
echo "preamble dir: $PDIR"
grep -E "^\[blocks\]" /tmp/${TAG}_preamble.log || { echo "no [blocks] partition lines: per-block partitioning failed"; exit 1; }

COMMON=(finn_s12_build.py "$PDIR" --tag "$TAG" --conv-order "$CONV_ORDER" --folding-json "$FOLDING" --blocks --partitions "${PARTS[@]}"
        --block-shapes "$SHAPES" --target-fps $TARGET_FPS --mvau-wwidth-max $WWIDTH)

echo "=== bridge dry run (folding + FIFO plan per block, no HLS / Vivado) ==="
python3 "${COMMON[@]}" --bridge-only 2>&1 | tee /tmp/${TAG}_bridge.log
grep -E "FIFO role bridge|MISMATCH|NO MATCH" /tmp/${TAG}_bridge.log || true
[ "$STEP" = "bridge" ] && exit 0

echo "=== standalone builds: HLS + IP stitch + rtlsim per block ==="
EXTRA=(--max-workers 4)
[ "$NO_OOC" = "1" ] && EXTRA+=(--no-ooc)
nohup python3 "${COMMON[@]}" "${EXTRA[@]}" > /tmp/${TAG}_build.log 2>&1 &
echo "$(date): LAUNCHED pid $!   tail -f /tmp/${TAG}_build.log"
echo "results: finn_deployment_outputs/${TAG}_milpfold_blocks*_<ts>/partition_<id>_<block>/ (partition<id>_${TAG}_milpfold_stitched.onnx, report/rtlsim_performance.json)"
