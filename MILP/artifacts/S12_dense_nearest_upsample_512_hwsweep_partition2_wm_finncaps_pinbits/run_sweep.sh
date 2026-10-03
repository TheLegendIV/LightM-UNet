#!/bin/bash
# MILP re-solve under FINN auto-fold's whole-network cost as hard caps (priced with finn_cost_model.py by
# MILP/utils/price_autofold_all_partitions.py on the 8 autofold_partitionN.onnx): LUT 80,598 (0.3498),
# BRAM18 509.4 (0.8164), DSP 451 (0.2610), FPS floor 190.7 (auto-fold's slowest node, partition 7).
# Bits PINNED to the fine-tuned _wm baseline_both_off bits (the bits FINN auto-fold was built on) -> folding-only test.
# Fractions rounded UP at 4 dp so FINN's own fold stays feasible. Bits free (4,6,8). Run in lightmunet_dev.
set -u
cd /workspace/LightM-UNet
OUT=MILP/artifacts/S12_dense_nearest_upsample_512_hwsweep_partition2_wm_finncaps_pinbits
COMMON="--config config_12_dense_relu_nearest_upsample_wm \
  --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_upsample_wm.json \
  --candidate-bits 4,6,8 --max-lut-fraction 0.3499 --max-bram-fraction 0.8164 --max-dsp-fraction 0.2611 \
  --max-uram-fraction 1.0 --force-dsp --target-fps 190.7 --time-limit 600 --gap-rel 0.02 \
  --pin-bits-file MILP/artifacts/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/baseline_both_off/layer_bits_folding_baseline_both_off.json"
run() { # tag extra-flags...
  tag=$1; shift
  mkdir -p $OUT/$tag
  python3 MILP/finn_milp.py $COMMON "$@" --out-file $OUT/$tag/layer_bits_folding_$tag.json > $OUT/$tag/solve.log 2>&1 &
}
run baseline_both_off
run dsrSweep_pbiOff_dsr1.5 --dsr-ratio 1.5
run dsrSweep_pbiOff_dsr3.0 --dsr-ratio 3.0
run dsrSweep_pbiOff_dsr7.5 --dsr-ratio 7.5
run dsrSweep_pbiOff_dsr15.0 --dsr-ratio 15.0
run dsrSweep_pbiOff_dsr150.0 --dsr-ratio 150.0
run pbiSweep_dsrOff_pbi1.5 --pbi-ratio 1.5
wait
