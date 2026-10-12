#!/bin/bash
# Uniform INT4 / INT6 / INT8, S12 dense bilinear-512 (config_12_dense_relu_warmstart150ep), --min-resources, 250 FPS @ 100 MHz, every resource cap at 1.0 of the device (LUT / BRAM / DSP / URAM).
# Stage 1 solves the three arms (MILP/finn_milp.py, ~1 min each), stage 2 simulates every OPTIMAL arm in the analytical block models (MILP/analytical/net_explicit.py): FIFOs sized to remove deadlocks
# only, each block's rate compared with its MILP cycles (> 5 % is a WARNING), whole net chained. Writes <arm>/ (folding json, summary.csv, solve.log) and <arm>_simfifo/ (simulated FIFO lists, run.log).
# Run in lightmunet_dev:   bash MILP/artifacts/S12_dense_512_uniform_cap1_v1/run_arms.sh [solve|sim|all] [arm ...]       (default: all three arms; sim ~1-3 h, arms in parallel, --workers 2 each)
set -u
cd /workspace/LightM-UNet
HERE=MILP/artifacts/S12_dense_512_uniform_cap1_v1
CONFIG=config_12_dense_relu_warmstart150ep
STAGE=${1:-all}; shift || true
ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then ARMS=(uniform_int4 uniform_int6 uniform_int8); fi

solve() {
  local arm=$1 bits=${1#uniform_int}
  mkdir -p "$HERE/$arm"
  python3 MILP/finn_milp.py --config $CONFIG --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_warmstart150ep.json \
    --candidate-bits "$bits" --force-dsp --min-resources --max-lut-fraction 1.0 --max-bram-fraction 1.0 --max-dsp-fraction 1.0 --max-uram-fraction 1.0 \
    --target-fps 250 --clock-mhz 100 --time-limit 900 --gap-rel 0.001 --out-file "$HERE/$arm/layer_bits_folding_$arm.json" > "$HERE/$arm/solve.log" 2>&1
}
simulate() {
  local arm=$1
  mkdir -p "$HERE/${arm}_simfifo"
  python3 MILP/analytical/net_explicit.py "$HERE/$arm/layer_bits_folding_$arm.json" --out-dir "$HERE/${arm}_simfifo" --tag "${arm}_simfifo" --config $CONFIG --workers 2 ${NE_ARGS:-} > "$HERE/${arm}_simfifo/run.log" 2>&1
}
status() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$HERE/$1/layer_bits_folding_$1.json" 2>/dev/null || echo "none"; }

if [ "$STAGE" = solve ] || [ "$STAGE" = all ]; then
  pids=(); for arm in "${ARMS[@]}"; do solve "$arm" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p"; done
  for arm in "${ARMS[@]}"; do echo "$arm: $(status "$arm")"; done
fi
if [ "$STAGE" = sim ] || [ "$STAGE" = all ]; then
  pids=(); run=()
  for arm in "${ARMS[@]}"; do
    if [ "$(status "$arm")" = Optimal ]; then simulate "$arm" & pids+=($!); run+=("$arm"); else echo "$arm: not Optimal, not simulated"; fi
  done
  rc=0
  for i in "${!pids[@]}"; do if wait "${pids[$i]}"; then echo "${run[$i]}: sim ok"; else echo "${run[$i]}: sim exit $? (see $HERE/${run[$i]}_simfifo/run.log)"; rc=1; fi; done
  exit $rc
fi
