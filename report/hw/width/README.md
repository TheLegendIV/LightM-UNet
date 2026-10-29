# Half (1/2) vs quarter (1/4) channel width, S12 dense / ReLU / bilinear / 512

Half width = `CHANNELS (8,32,64,32,8)` (`ILP/configs/config_12_dense_relu_warmstart150ep_w32_64.py`, cost only, no checkpoint); quarter = the hardware baseline `(4,16,32,16,4)`. Everything else identical.

Constraints copied from `ILP/artifacts/S12_512_ablation_v2`: uniform INT4/6/8, `--force-dsp --min-resources`, 150 FPS @ 100 MHz, 300 s / gap 0.001; `relaxed_fps150` = every cap 1.0, `cap40_fps150` = every cap 0.4. Solved with the current cost model (`run_arms.sh`, `summarize.py` -> `width_cost_summary.csv`). The quarter INT6 cap40 arm reproduces the stored ablation v2 artifact exactly (91,779 LUT). MILP estimates only: no FIFO simulation, no hardware validation.

`fps190/`: first attempt at 250 FPS was infeasible for BOTH widths (current model: `final.argmax` = 524,288 cycles/frame, measured 2026-10-16, caps a 512x512 design at ~190 FPS); `fps190/` is the same solve at 190 FPS, kept for reference.

## Max-FPS arms (`maxfps_caps1/`, `run_maxfps.sh`, `summarize_maxfps.py` -> `maxfps_summary.csv`)

`--max-fps` (pass 1: minimise the slowest node's cycles; pass 2: min-resources objective at that frame time + 0.1 % slack), every cap 1.0, uniform INT4/6/8, both widths. ALL six arms hit the same wall: pass-1 frame time 524,288 cycles = `final.argmax` (LabelSelect, 2 cycles/pixel at PE 5, measured 2026-10-16) = **190.7 FPS @ 100 MHz** (reported 190.5 after the pass-2 slack). The argmax has 5 labels at every width, so this ceiling says nothing about the width-dependent compute limit. Pass 2's objective is the equal-weight mean of the LUT/BRAM/DSP fractions, not LUT alone, so a higher-FPS arm can show fewer LUTs and more BRAM than a lower-FPS arm.
