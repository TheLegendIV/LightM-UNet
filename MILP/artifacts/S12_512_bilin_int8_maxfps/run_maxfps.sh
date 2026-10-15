#!/bin/bash
# S12 dense bilinear-512 INT8 (candidate bits 8 only), every resource cap 1.0, warmstart150ep config, 1200x40 sensitivity (|trace| x delta; irrelevant at one bit width), FIFO / DWC modelled, SWU in BRAM allowed.
#   A  maxfps/        --max-fps: pass 1 minimises the slowest node's frame cycles (= maximises FPS), pass 2 minimises resources at that frame time (+0.1 %).
#   B  pinned_minres/ --target-fps <the FPS A reached> --min-resources: the cheapest folding at exactly that FPS (the one the FIFO simulation is run on).
# Run in lightmunet_dev:   bash MILP/artifacts/S12_512_bilin_int8_maxfps/run_maxfps.sh
set -u
cd /workspace/LightM-UNet
HERE=MILP/artifacts/S12_512_bilin_int8_maxfps
COMMON="--config config_12_dense_relu_warmstart150ep --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_warmstart150ep_n1200p40_seed0.json --candidate-bits 8 --force-dsp --max-lut-fraction 1.0 --max-bram-fraction 1.0 --max-dsp-fraction 1.0 --max-uram-fraction 1.0 --clock-mhz 100 --time-limit 900 --gap-rel 0.001"
mkdir -p $HERE/maxfps $HERE/pinned_minres
python3 MILP/finn_milp.py $COMMON --max-fps --out-file $HERE/maxfps/layer_bits_folding_maxfps.json > $HERE/maxfps/solve.log 2>&1
FPS=$(python3 -c "import csv,sys; print(next(csv.DictReader(open(sys.argv[1])))['fps'])" $HERE/maxfps/summary.csv)
echo "A max-fps: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" $HERE/maxfps/layer_bits_folding_maxfps.json) FPS=$FPS"
python3 MILP/finn_milp.py $COMMON --target-fps $FPS --min-resources --out-file $HERE/pinned_minres/layer_bits_folding_pinned_minres.json > $HERE/pinned_minres/solve.log 2>&1
echo "B pinned min-resources @ $FPS FPS: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" $HERE/pinned_minres/layer_bits_folding_pinned_minres.json)"
