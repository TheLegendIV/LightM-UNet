#!/bin/bash
# Arm C WITHOUT DSR: same bits as lex_dsr2_fps305 (pinned, epsilon 0), pass-2 objective (--min-resources), same
# FPS floor 305.17, same width cap 80, same resource roof (= pass 1's LUT/BRAM/DSP use in lex_dsr2_fps305) --
# only --dsr-ratio is dropped. Isolates the DSR constraint from the MILP folding search itself.
# Usage (in lightmunet_dev): run_c_nodsr.sh
set -u
cd /workspace/LightM-UNet
SRC=MILP/artifacts/S12_dense_arms_bc_v1/lex_dsr2_fps305
OUT=MILP/artifacts/S12_dense_arms_bc_v1/lex_nodsr_fps305
mkdir -p $OUT
read LUT BRAM DSP < <(python3 - <<PY
import json
d=json.load(open("$SRC/stage1_layer_bits_folding_lex_dsr2_fps305.json"))["_diagnostics"]
f=1+1e-6
print(d["lut_pct_of_budget"]/100*f, d["bram_pct_of_budget"]/100*f, d["dsp_pct_of_budget"]/100*f)
PY
)
python3 MILP/finn_milp.py --config config_12_dense_relu_nearest_upsample_wm \
  --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_upsample_wm.json \
  --candidate-bits 4,6,8 --pin-bits-file $SRC/layer_bits_folding_lex_dsr2_fps305.json \
  --max-lut-fraction $LUT --max-bram-fraction $BRAM --max-dsp-fraction $DSP \
  --max-uram-fraction 1.0 --force-dsp --target-fps 305.17 --mvau-wwidth-max 80 \
  --min-resources --time-limit 900 --gap-rel 0.005 \
  --out-file $OUT/layer_bits_folding_lex_nodsr_fps305.json > $OUT/solve.log 2>&1
