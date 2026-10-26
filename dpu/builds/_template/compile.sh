#!/usr/bin/env bash
# Compile the quantized xmodel for the DPU on the board. Runs inside the Vitis AI container.
# The class map (argmax) is produced on the DPU: add_argmax.py appends an XIR argmax first.
set -euo pipefail

XMODEL="$(ls /workspace/quant/*_int.xmodel | head -n1)"
ARCH="${ARCH_JSON:?ARCH_JSON not set}"
[[ -f "$ARCH" ]] || ARCH="/workspace/inputs/arch/$(basename "$(dirname "$ARCH")")/arch.json"

python3 /job/add_argmax.py "$XMODEL" /workspace/quant/argmax_int.xmodel
XMODEL=/workspace/quant/argmax_int.xmodel

echo "xmodel: $XMODEL"
echo "arch:   $ARCH"
grep -i fingerprint "$ARCH" || true   # must equal `xdputil query` on the board

vai_c_xir -x "$XMODEL" -a "$ARCH" -o /workspace/compiled -n "${JOB}"

# Check the log for: DPU subgraph count, ops left on CPU, tensor shapes.
