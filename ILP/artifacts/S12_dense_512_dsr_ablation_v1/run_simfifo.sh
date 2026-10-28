#!/bin/bash
# Simulate every MILP arm of the 512 DSR ablation in the analytical block models + simulators and grow its FIFOs until nothing deadlocks
# (ILP/analytical/net_explicit.py). Writes <arm>_simfifo/layer_bits_folding_<arm>_simfifo.json next to the base arm: same folding, FIFO lists replaced by the simulated ones.
# Run in lightmunet_dev from anywhere:   bash ILP/artifacts/S12_dense_512_dsr_ablation_v1/run_simfifo.sh [arm ...]     (default: the first arm of each of the three distinct foldings)
# All arms run in parallel (--workers per arm for the block simulations); the whole-net chain run of each arm is single threaded (minutes per round).
set -u
cd /workspace/LightM-UNet
HERE=ILP/artifacts/S12_dense_512_dsr_ablation_v1
ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then ARMS=(dsr_50pct); fi   # one per DISTINCT folding (arm_groups.json): dsr_100pct/200pct/off share it (1-25 % are infeasible, no simulation)
pids=()
for arm in "${ARMS[@]}"; do
  mkdir -p "$HERE/${arm}_simfifo"
  python3 ILP/analytical/net_explicit.py "$HERE/$arm/layer_bits_folding_$arm.json" --out-dir "$HERE/${arm}_simfifo" --tag "${arm}_simfifo" \
    --config config_12_dense_relu_nearest_upsample --workers 2 > "$HERE/${arm}_simfifo/run.log" 2>&1 &
  pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "${ARMS[$i]}: ok"; else echo "${ARMS[$i]}: exit $? (block or chain did not pass: see $HERE/${ARMS[$i]}_simfifo/run.log)"; rc=1; fi
done
exit $rc
