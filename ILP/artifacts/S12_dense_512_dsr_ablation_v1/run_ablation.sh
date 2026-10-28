#!/bin/bash
# DSR ablation, S12 dense nearest-upsample (NO conv after the upsample) ReLU net, 512x512, uniform INT6, 250 fps (every node <= 400,000 cycles/frame), FINN mvau_wwidth_max 72,
# FIFO / DWC modelling on, objective = minimise resources (mean of LUT / BRAM / DSP fractions); no DSR floor.
#
# DSR = "downstream no slower than upstream" in cycles per frame, on every compute node (conv / MVAU layers, pad-MVAU, argmax) against its nearest compute ancestor:
#   cycles[C] <= (1 + pct/100) * cycles[P]     (no floor; the cap stays F = 400,000)
# Arms:  dsr_{1,5,10,25,50,100,200}pct : --dsr-pct P     dsr_off : --dsr-pct none (no rate rule)
# Floor 0 keeps F (400,000) as the only absolute cap (no hidden fps / floor target). There is NO exemption for nodes that cannot satisfy the rule: a pct for which the chain cannot hold is INFEASIBLE
# and stays in the table as such (512: regular5.0.conv tops out at 196,608 cycles, final cannot go below ~263,171 -> infeasible below ~34 %).
# Hardware-only variants and the FINN auto-fold control are not MILP runs: dsr_off_finnfifo (summarize_arms.py) and dsr_ablation_finn_autofold (dataflow/builds/S12_dense_512_dsr_ablation_v1/).
# Run in lightmunet_dev (all arms in parallel):  bash ILP/artifacts/S12_dense_512_dsr_ablation_v1/run_ablation.sh ; then  python3 ILP/artifacts/S12_dense_512_dsr_ablation_v1/summarize_arms.py
set -u
cd /workspace/LightM-UNet
OUT=ILP/artifacts/S12_dense_512_dsr_ablation_v1
source $OUT/arms_config.sh                        # CONFIG BITS FPS CLOCK_MHZ WWIDTH: shared by every arm
# the sensitivity file is only required by the CLI: bits are fixed (one --candidate-bits value), so it does not influence the solve
COMMON="--config $CONFIG   --sensitivity-file ILP/artifacts/layer_sensitivity_12_dense_relu_nearest_conv_upsample_256.json   --candidate-bits $BITS --max-lut-fraction 1.0 --max-bram-fraction 1.0 --max-dsp-fraction 1.0   --max-uram-fraction 1.0 --force-dsp --target-fps $FPS --clock-mhz $CLOCK_MHZ --min-resources   --mvau-wwidth-max $WWIDTH --time-limit 600 --gap-rel 0.001"
run() { # tag extra-flags...
  tag=$1; shift
  mkdir -p $OUT/$tag
  python3 ILP/finn_milp.py $COMMON "$@" --out-file $OUT/$tag/layer_bits_folding_$tag.json > $OUT/$tag/solve.log 2>&1 &
}
for p in 1 5 10 25 50 100 200; do run dsr_${p}pct --dsr-pct $p; done
run dsr_off --dsr-pct none
wait
