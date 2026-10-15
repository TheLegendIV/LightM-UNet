#!/bin/bash
# S12 dense bilinear-512 INT8 (candidate bits 8 only: all-8-bit was the accuracy optimum), 300 FPS @ 100 MHz, caps LUT 0.7 / BRAM 0.8 / DSP 0.8, --min-resources (fixed bits -> cheapest folding),
# with the downstream-rate rule --dsr-pct tried at 25 %, then 35 %, then 50 % (stops at the first feasible one). SWU line buffer in BRAM allowed (default), FIFO / DWC modelled.
# Run in lightmunet_dev:   bash MILP/artifacts/S12_512_bilin_int8_dsr_v1/run_dsr.sh
set -u
cd /workspace/LightM-UNet
HERE=MILP/artifacts/S12_512_bilin_int8_dsr_v1
for pct in 25 35 50; do
  d=$HERE/dsr$pct; mkdir -p $d
  python3 MILP/finn_milp.py --config config_12_dense_relu_warmstart150ep --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_warmstart150ep_n1200p40_seed0.json \
    --candidate-bits 8 --force-dsp --min-resources --max-lut-fraction 0.7 --max-bram-fraction 0.8 --max-dsp-fraction 0.8 --max-uram-fraction 1.0 \
    --target-fps 300 --clock-mhz 100 --dsr-pct $pct --time-limit 900 --gap-rel 0.001 --out-file $d/layer_bits_folding_dsr$pct.json > $d/solve.log 2>&1
  st=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" $d/layer_bits_folding_dsr$pct.json 2>/dev/null || echo none)
  echo "dsr-pct $pct: $st"
  [ "$st" = Optimal ] && break
done
