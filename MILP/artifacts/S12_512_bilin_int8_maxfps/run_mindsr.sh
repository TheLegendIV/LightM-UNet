#!/bin/bash
# Same setup as run_maxfps.sh (INT8, caps 1.0, FPS pinned at the maximum A reached), now with the DSR objective:
#   C  mindsr/               --target-fps <max FPS> --min-stall: pass 1 minimises the total stall (sum over DSR edges of max(0, cycles[C] - cycles[P])), pass 2 minimises resources at that stall (+0.1 %).
#   D  mindsr_pinned_minres/ --target-fps <max FPS> --dsr-pct <C's achieved worst-edge DSR, rounded up> --min-resources: the cheapest folding whose worst edge is no worse than C's.
# Run in lightmunet_dev (after run_maxfps.sh):   bash MILP/artifacts/S12_512_bilin_int8_maxfps/run_mindsr.sh
set -u
cd /workspace/LightM-UNet
HERE=MILP/artifacts/S12_512_bilin_int8_maxfps
COMMON="--config config_12_dense_relu_warmstart150ep --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_warmstart150ep_n1200p40_seed0.json --candidate-bits 8 --force-dsp --max-lut-fraction 1.0 --max-bram-fraction 1.0 --max-dsp-fraction 1.0 --max-uram-fraction 1.0 --clock-mhz 100 --time-limit 900 --gap-rel 0.001"
FPS=$(python3 -c "import csv,sys; print(next(csv.DictReader(open(sys.argv[1])))['fps'])" $HERE/maxfps/summary.csv)
mkdir -p $HERE/mindsr $HERE/mindsr_pinned_minres
python3 MILP/finn_milp.py $COMMON --target-fps $FPS --min-stall --out-file $HERE/mindsr/layer_bits_folding_mindsr.json > $HERE/mindsr/solve.log 2>&1
echo "C min-dsr @ $FPS FPS: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" $HERE/mindsr/layer_bits_folding_mindsr.json)"
PCT=$(python3 -c "import csv,math,sys; print(f'{math.ceil(float(next(csv.DictReader(open(sys.argv[1])))[\"dsr_achieved_pct\"])*100)/100:.2f}')" $HERE/mindsr/summary.csv)
python3 MILP/finn_milp.py $COMMON --target-fps $FPS --dsr-pct $PCT --min-resources --out-file $HERE/mindsr_pinned_minres/layer_bits_folding_mindsr_pinned_minres.json > $HERE/mindsr_pinned_minres/solve.log 2>&1
echo "D dsr-pct $PCT + min-resources: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" $HERE/mindsr_pinned_minres/layer_bits_folding_mindsr_pinned_minres.json)"
