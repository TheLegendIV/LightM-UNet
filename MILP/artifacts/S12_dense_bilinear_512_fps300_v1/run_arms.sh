#!/bin/bash
# S12 dense bilinear-512 (config_12_dense_relu_warmstart150ep), $FPS (default 300) FPS @ $CLOCK (default 100) MHz (the maxpool floor is ~305 FPS at 100 MHz; 350+ is infeasible).
# --lexicographic: pass 1 = accuracy objective (least sensitivity), pass 2 = bits pinned, minimise resources under pass 1's own use.
#   bits468 / bits68       : candidate bits 4,6,8 / 6,8; caps LUT 0.7 / BRAM 0.9 / DSP 0.9 (URAM 1.0, inert)
#   bits468_r2 / bits68_r2 : round 2 -- same bits, but the caps are the RESOURCES OF THE BASE ARM'S PASS-2 RESULT (summary.csv lut/bram/dsp_pct_of_budget, rounded UP to 4 decimals so
#                            that result stays feasible): max accuracy under those caps, then min resources again.
#   bits468_refine / bits68_refine : the ORIGINAL caps (0.7 / 0.9 / 0.9), --refine-bits: pass 1 accuracy, pass 2 min resources, then pass 3 = pass 2's folding pinned, bits freed, accuracy again under the
#                            same caps (the budget pass 2 shaved off is room for the bits). stage1_ / stage2_ files hold passes 1 / 2.
# Run in lightmunet_dev:   bash MILP/artifacts/S12_dense_bilinear_512_fps300_v1/run_arms.sh [arm ...]     (default: bits468 bits68; round 2 arms must be named, after their base arms are Optimal)
set -u
cd /workspace/LightM-UNet
FPS=${FPS:-300}; CLOCK=${CLOCK:-100}
HERE=MILP/artifacts/S12_dense_bilinear_512_fps${FPS}_v1
CONFIG=config_12_dense_relu_warmstart150ep
ARMS=("$@"); if [ ${#ARMS[@]} -eq 0 ]; then ARMS=(bits468 bits68); fi
solve() {
  local arm=$1 base=${1%_r2} lut=0.7 bram=0.9 dsp=0.9 bits refine=""
  base=${base%_refine}; [ "$arm" != "${arm%_refine}" ] && refine="--refine-bits"
  bits=$(echo "${base#bits}" | sed 's/./&,/g; s/,$//')       # 468 -> 4,6,8 ; 68 -> 6,8
  if [ "$arm" != "${arm%_r2}" ]; then
    read lut bram dsp < <(python3 - "$HERE/$base/summary.csv" <<'P'
import csv, math, sys
r = next(csv.DictReader(open(sys.argv[1])))
print(*(f"{math.ceil(float(r[k]) / 100 * 1e4) / 1e4:.4f}" for k in ("lut_pct_of_budget", "bram_pct_of_budget", "dsp_pct_of_budget")))
P
)
  fi
  mkdir -p "$HERE/$arm"
  echo "$arm: bits $bits caps LUT $lut BRAM $bram DSP $dsp" > "$HERE/$arm/solve.log"
  python3 MILP/finn_milp.py --config $CONFIG --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_warmstart150ep.json \
    --candidate-bits "$bits" --force-dsp --lexicographic $refine --max-lut-fraction $lut --max-bram-fraction $bram --max-dsp-fraction $dsp --max-uram-fraction 1.0 \
    --target-fps $FPS --clock-mhz $CLOCK --time-limit 900 --gap-rel 0.001 --out-file "$HERE/$arm/layer_bits_folding_$arm.json" >> "$HERE/$arm/solve.log" 2>&1
}
pids=(); for arm in "${ARMS[@]}"; do solve "$arm" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
for arm in "${ARMS[@]}"; do echo "$arm: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$HERE/$arm/layer_bits_folding_$arm.json" 2>/dev/null || echo none)"; done
