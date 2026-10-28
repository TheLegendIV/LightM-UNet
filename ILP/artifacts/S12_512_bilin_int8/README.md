# S12_512_bilin_int8

MILP solve of the S12 dense **bilinear-decoder 512x512** net (`config_12_dense_relu_warmstart150ep`), stored from
`S12_dense_bilinear_512_fps300_v1_sens1200_caps70_80_80/bits468` (candidate bits 4,6,8; the 6,8 arm `bits68` chose the same bits).

* Sensitivity: `layer_sensitivity_12_dense_relu_warmstart150ep_n1200p40_seed0.json` (1200 batches x 40 probes), |trace| x delta, objective scaled x1e6.
* 300 FPS @ 100 MHz (350 / 325 FPS infeasible: the unfoldable initial StreamingMaxPool needs 327,680 cycles/frame -> 305 FPS at most), caps LUT 0.7 / BRAM 0.8 / DSP 0.8,
  `--lexicographic`: pass 1 accuracy, pass 2 minimise resources. SWU line buffer in BRAM allowed (6 layers took it).
* Result: all 85 weight layers and all 85 activation sites at 8 bits (hence "int8"), LUT 51.6 % / BRAM 39.1 % / DSP 17.0 % after pass 2, 305.2 FPS.
* `S12_512_bilin_int8_simfifo/` = the same folding with the FIFOs sized by `ILP/analytical/net_explicit.py`.
