#!/bin/bash
# DSR ablation, S12 dense nearest-upsample (NO conv after the upsample) ReLU net, 256x256, uniform INT6, 250 fps (every node <= 400,000 cycles/frame),
# FINN mvau_wwidth_max 72, FIFO / DWC modelling on, objective = minimise resources (mean of LUT / BRAM / DSP fractions).
#
# DSR = "downstream no slower than upstream" in cycles per frame, on every compute node (conv / MVAU layers, pad-MVAU, argmax) against its nearest compute ancestor:
#   cycles[C] <= max(floor * F, (1 + pct/100) * cycles[P]),   F = 400,000.
# Arms:
#   ratchet_{1,25,100,200}pct : --dsr-pct P --dsr-floor 0.33
#   ratchet_off             : --dsr-pct none   (no rate rule; the element-rate DSR is off by default)
# Why floor 0.33 (= 132,000 cycles): it is the smallest round floor that is FEASIBLE for every arm. With floor 0 the DSR is infeasible below +50% (tested: infeasible at 1/5/12/25/33%, feasible at 50% and
# up, with or without the width cap): the last blocks have tiny layers (regular5: 1 -> 4 channels, 65,536 cycles of work at most) in front of the 4x-work final transposed conv, so no strict "no slower"
# chain can hold without a floor. From floor ~0.4 up the DSR never binds on this design (it sits at ~147k cycles) and the small-pct arms collapse to one folding; at 0.33 it binds.
#
# The FINN auto-fold control (tag ratchet_ablation_finn_autofold) is NOT a MILP arm: it is built on hardware, see hardware/builds/S12_dense_256_ratchet_ablation_v1/.
# --mvau-wwidth-max 72 must equal the value given to FINN's auto-fold (finn_s12_build.py --mvau-wwidth-max 72); 72 = 12 x 6-bit weights per PE.
# Run in lightmunet_dev (all arms in parallel):  bash MILP/artifacts/S12_dense_256_ratchet_ablation_v1/run_ablation.sh ; then  python3 MILP/artifacts/S12_dense_256_ratchet_ablation_v1/summarize_arms.py
set -u
cd /workspace/LightM-UNet
OUT=MILP/artifacts/S12_dense_256_ratchet_ablation_v1
source $OUT/arms_config.sh                        # CONFIG BITS FPS CLOCK_MHZ DSR_FLOOR WWIDTH: shared by every arm incl. the analytical one
# the sensitivity file is only required by the CLI: bits are fixed (one --candidate-bits value), so it does not influence the solve
COMMON="--config $CONFIG   --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_conv_upsample_256.json   --candidate-bits $BITS --max-lut-fraction 1.0 --max-bram-fraction 1.0 --max-dsp-fraction 1.0   --max-uram-fraction 1.0 --force-dsp --target-fps $FPS --clock-mhz $CLOCK_MHZ --min-resources   --mvau-wwidth-max $WWIDTH --time-limit 600 --gap-rel 0.001"
run() { # tag extra-flags...
  tag=$1; shift
  mkdir -p $OUT/$tag
  python3 MILP/finn_milp.py $COMMON "$@" --out-file $OUT/$tag/layer_bits_folding_$tag.json > $OUT/$tag/solve.log 2>&1 &
}
for p in 1 25 100 200; do run ratchet_${p}pct --dsr-pct $p --dsr-floor $DSR_FLOOR; done
run ratchet_off --dsr-pct none
wait
