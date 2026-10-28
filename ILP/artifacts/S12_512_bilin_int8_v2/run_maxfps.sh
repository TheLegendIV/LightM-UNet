#!/bin/bash
# S12 dense bilinear-512 INT8 (candidate bits 8 only), every resource cap 1.0, warmstart150ep config, 1200x40 sensitivity (|trace| x delta; irrelevant at one bit width), FIFO / DWC modelled,
# SWU in BRAM allowed. v2 = the model corrected against the real INT8 build (2026-10-16): LabelSelect 2 cycles/fold/pixel (PE a real choice), StreamingMaxPool 6.5 cycles per 2x2 window
# iteration, InferPool (depthwise SWG + Pool_hls) as a per-pool choice (initial pool only: --pool-impl-scope initial), clock = the board's PL clock (187.481262 MHz).
#   maxfps/  --max-fps: pass 1 minimises the slowest node's frame cycles (= maximises FPS), pass 2 minimises resources at that frame time (+0.1 %).
# Run in the repo root (host venv or lightmunet_dev):   bash ILP/artifacts/S12_512_bilin_int8_v2/run_maxfps.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
HERE=ILP/artifacts/S12_512_bilin_int8_v2
mkdir -p $HERE/maxfps
python3 ILP/finn_milp.py --config config_12_dense_relu_warmstart150ep \
  --sensitivity-file ILP/artifacts/layer_sensitivity_12_dense_relu_warmstart150ep_n1200p40_seed0.json \
  --candidate-bits 8 --force-dsp --max-lut-fraction 1.0 --max-bram-fraction 1.0 --max-dsp-fraction 1.0 --max-uram-fraction 1.0 \
  --clock-mhz 187.481262 --time-limit 900 --gap-rel 0.001 --max-fps --pool-impl choose --pool-impl-scope initial \
  --out-file $HERE/maxfps/layer_bits_folding_maxfps.json > $HERE/maxfps/solve.log 2>&1
echo "max-fps: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" $HERE/maxfps/layer_bits_folding_maxfps.json) FPS=$(python3 -c "import csv,sys; print(next(csv.DictReader(open(sys.argv[1])))['fps'])" $HERE/maxfps/summary.csv)"
