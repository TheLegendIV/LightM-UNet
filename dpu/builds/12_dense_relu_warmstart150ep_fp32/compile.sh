#!/usr/bin/env bash
# Compile the quantized xmodel for the DPU on the board. Runs inside the Vitis AI container.
# The class map (argmax) is produced on the DPU: add_argmax.py appends an XIR argmax first.
#
#   QUANT_DIR / COMPILE_OUT / COMPILE_NAME select the PTQ (default) or QAT model:
#   QUANT_DIR=/workspace/quant_qat COMPILE_OUT=/workspace/compiled_qat COMPILE_NAME=${JOB}_qat
set -euo pipefail

QUANT_DIR="${QUANT_DIR:-/workspace/quant}"
COMPILE_OUT="${COMPILE_OUT:-/workspace/compiled}"
COMPILE_NAME="${COMPILE_NAME:-${JOB}}"

XMODEL="$(ls "$QUANT_DIR"/*_int.xmodel | grep -v argmax | head -n1)"
ARCH="${ARCH_JSON:?ARCH_JSON not set}"
[[ -f "$ARCH" ]] || ARCH="/workspace/inputs/arch/$(basename "$(dirname "$ARCH")")/arch.json"

python3 /job/add_argmax.py "$XMODEL" "$QUANT_DIR/argmax_int.xmodel"
XMODEL="$QUANT_DIR/argmax_int.xmodel"

echo "xmodel: $XMODEL"
echo "arch:   $ARCH"
grep -i fingerprint "$ARCH" || true   # must equal `xdputil query` on the board

vai_c_xir -x "$XMODEL" -a "$ARCH" -o "$COMPILE_OUT" -n "$COMPILE_NAME"

# Check the log for: DPU subgraph count, ops left on CPU, tensor shapes.
