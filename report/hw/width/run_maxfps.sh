#!/bin/bash
# Half (1/2) vs quarter (1/4) channel width of the S12 dense / ReLU / bilinear / 512 hardware baseline, uniform INT4/6/8, priced with ILP/finn_cost_model.py through the MILP.
# Constraints copied from ILP/artifacts/S12_512_ablation_v2 (relaxed_fps150 = caps 1.0, cap40_fps150 = caps 0.4): --force-dsp --min-resources, 150 FPS @ 100 MHz, time limit 300 s, gap 0.001.
# Both widths re-solved with the CURRENT cost model. Sensitivity file is irrelevant under with one candidate bit width (site names do not depend on width).
# 250 FPS is infeasible with the current model at 512x512 (final.argmax = 524,288 cycles, measured 2026-10-16 -> max ~190 FPS).
# MAX-FPS variant of run_arms.sh: objective = maximum FPS (pass 1), then min resources at that frame time (pass 2); caps 1.0, no --target-fps.
# Run in lightmunet_dev:   bash report/hw/width/run_maxfps.sh
set -u
cd /workspace/LightM-UNet
CAP=1.0; HERE=report/hw/width/maxfps_caps1
declare -A CFG=( [q]=config_12_dense_relu_warmstart150ep [h]=config_12_dense_relu_warmstart150ep_w32_64 )
pids=()
for w in q h; do for b in 4 6 8; do
  d=$HERE/${w}_int$b; mkdir -p $d
  python3 ILP/finn_milp.py --config ${CFG[$w]} --sensitivity-file ILP/artifacts/layer_sensitivity_12_dense_relu_warmstart150ep_n1200p40_seed0.json \
    --candidate-bits $b --force-dsp --max-lut-fraction $CAP --max-bram-fraction $CAP --max-dsp-fraction $CAP --max-uram-fraction $CAP \
    --max-fps --clock-mhz 100 --time-limit 300 --gap-rel 0.001 --out-file $d/layer_bits_folding.json > $d/solve.log 2>&1 &
  pids+=($!)
done; done
for p in "${pids[@]}"; do wait $p; done
for w in q h; do for b in 4 6 8; do echo "${w}_int$b: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" $HERE/${w}_int$b/layer_bits_folding.json 2>/dev/null || echo none)"; done; done
