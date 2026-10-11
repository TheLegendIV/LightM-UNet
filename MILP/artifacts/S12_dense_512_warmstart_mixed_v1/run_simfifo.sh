#!/bin/bash
# Simulate the three feasible foldings of this comparison (uniform INT4, uniform INT6, mixed 4/6/8; uniform INT8 is infeasible under the 0.5 cap once FIFOs are priced) in the analytical block models +
# simulators and grow the FIFOs until nothing deadlocks (MILP/analytical/net_explicit.py). Writes <arm>_simfifo/layer_bits_folding_<arm>_simfifo.json next to the base arm: same folding, FIFO lists replaced by
# the simulated ones. Pass = no deadlock and a whole-net period <= 400,000 cycles/frame.
# The analytical block models take ONE bit width for the whole net (no per-layer bits): the mixed folding is simulated at --bits 8, the LARGEST candidate. Cycles, stalls and FIFO DEPTHS (words) do not depend on
# bits; only stream widths do (BRAM aspect / LUT of the FIFOs, conservative at 8 bit). The MILP graph has no bilinear depthwise node, so these simulate the nearest-style up blocks.
# Run in lightmunet_dev from anywhere:   bash MILP/artifacts/S12_dense_512_warmstart_mixed_v1/run_simfifo.sh [arm ...]     (default: the three arms, in parallel, --workers 2 each; ~1-3 h)
set -u
cd /workspace/LightM-UNet
HERE=MILP/artifacts/S12_dense_512_warmstart_mixed_v1
declare -A BITS=([uniform_int4]=4 [uniform_int6]=6 [mixed_468]=8)
ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then ARMS=(uniform_int4 uniform_int6 mixed_468); fi
pids=()
for arm in "${ARMS[@]}"; do
  mkdir -p "$HERE/${arm}_simfifo"
  python3 MILP/analytical/net_explicit.py "$HERE/$arm/layer_bits_folding_$arm.json" --out-dir "$HERE/${arm}_simfifo" --tag "${arm}_simfifo" \
    --config config_12_dense_relu_warmstart150ep --bits ${BITS[$arm]} --workers 2 --slack 0.03 --max-rounds 2 > "$HERE/${arm}_simfifo/run.log" 2>&1 &
  pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "${ARMS[$i]}: ok"; else echo "${ARMS[$i]}: exit $? (block or chain did not pass: see $HERE/${ARMS[$i]}_simfifo/run.log)"; rc=1; fi
done
exit $rc
