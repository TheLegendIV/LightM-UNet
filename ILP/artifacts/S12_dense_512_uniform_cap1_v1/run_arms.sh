#!/bin/bash
# S12 dense bilinear-512 (config_12_dense_relu_warmstart150ep), 250 FPS @ 100 MHz. Four arms:
#   uniform_int4 / uniform_int6 / uniform_int8   --min-resources, every resource cap 1.0 of the device (LUT / BRAM / DSP / URAM)
#   mixed_68                                     bits 6 or 8 per layer, the accuracy objective (--lexicographic: least sensitivity, then least resources) with the 0.5 caps of the earlier
#                                                mixed_468 arm -- the caps are what forces a mix (all-INT8 needs 55 % LUT, all-INT6 44.8 %)
# Stage 1 solves the arms (ILP/finn_milp.py, ~1 min each), stage 2 simulates every OPTIMAL arm in the analytical block models (ILP/analytical/net_explicit.py): the sizing STARTS from the MILP's
# FIFO depths, may raise or lower any FIFO to reach each block's MILP cycles (<= +500 % of the starting depth), the whole net is chained. A block beyond 5 % of its MILP cycles is a WARNING.
# Writes <arm>/ (folding json, summary.csv, solve.log) and <arm>_simfifo/ (simulated FIFO lists, run.log).
# Run in lightmunet_dev:   bash ILP/artifacts/S12_dense_512_uniform_cap1_v1/run_arms.sh [solve|sim|all] [arm ...]       (default: all four arms; sim ~0.5-1 h, arms in parallel, --workers 2 each)
set -u
cd /workspace/LightM-UNet
HERE=ILP/artifacts/S12_dense_512_uniform_cap1_v1
CONFIG=config_12_dense_relu_warmstart150ep
STAGE=${1:-all}; shift || true
ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then ARMS=(uniform_int4 uniform_int6 uniform_int8 mixed_68); fi

solve() {
  local arm=$1 bits=${1#uniform_int} obj="--min-resources" cap=1.0
  if [ "$arm" = mixed_68 ]; then bits="6,8"; obj="--lexicographic"; cap=0.5; fi
  mkdir -p "$HERE/$arm"
  python3 ILP/finn_milp.py --config $CONFIG --sensitivity-file ILP/artifacts/layer_sensitivity_12_dense_relu_warmstart150ep.json \
    --candidate-bits "$bits" --force-dsp $obj --max-lut-fraction $cap --max-bram-fraction $cap --max-dsp-fraction $cap --max-uram-fraction $cap \
    --target-fps 250 --clock-mhz 100 --time-limit 900 --gap-rel 0.001 --out-file "$HERE/$arm/layer_bits_folding_$arm.json" > "$HERE/$arm/solve.log" 2>&1
}
simulate() {
  local arm=$1
  mkdir -p "$HERE/${arm}_simfifo"
  python3 ILP/analytical/net_explicit.py "$HERE/$arm/layer_bits_folding_$arm.json" --out-dir "$HERE/${arm}_simfifo" --tag "${arm}_simfifo" --config $CONFIG --workers 2 ${NE_ARGS:-} > "$HERE/${arm}_simfifo/run.log" 2>&1
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
