#!/bin/bash
# TEST: MILP built from the 300-epoch checkpoint's sensitivity (MILP/artifacts/layer_sensitivity_12_dense_relu_300ep_n64p30_seed0.json: 0 / 85 negative trace_w, 3 negative trace_a), bilinear-512 net
# (config_12_dense_relu_300ep = same architecture as warmstart150ep), candidate bits 4,6,8, 300 FPS @ 100 MHz, every cap 1.0, accuracy objective only (no --lexicographic / --min-resources).
# Run in lightmunet_dev:   bash MILP/artifacts/S12_dense_bilinear_512_sens300ep_test_v1/run_arms.sh
set -u
cd /workspace/LightM-UNet
HERE=MILP/artifacts/S12_dense_bilinear_512_sens300ep_test_v1/bits468_cap1
python3 MILP/finn_milp.py --config config_12_dense_relu_300ep --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_300ep_n64p30_seed0.json \
  --candidate-bits 4,6,8 --force-dsp --max-lut-fraction 1.0 --max-bram-fraction 1.0 --max-dsp-fraction 1.0 --max-uram-fraction 1.0 \
  --target-fps 300 --clock-mhz 100 --time-limit 900 --gap-rel 0.001 --out-file "$HERE/layer_bits_folding_bits468_cap1.json" > "$HERE/solve.log" 2>&1
echo "status: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$HERE/layer_bits_folding_bits468_cap1.json")"
