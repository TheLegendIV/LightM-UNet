#!/bin/bash
# The ONE definition of the target shared by every arm of this ablation (MILP arms, the FINN-autosized-FIFO variant, the FINN auto-fold control on hardware). Sourced by run_ablation.sh;
# summarize_arms.py checks that every result file carries exactly these values.
CONFIG=config_12_dense_relu_nearest_upsample       # 512x512 S12 dense nearest-upsample (noconv) ReLU net (the MILP has no bilinear depthwise node: that net exists only in the analytical flow)
BITS=6                                             # uniform weight = activation bits
FPS=250                                            # every node <= clock / FPS = 400,000 cycles per frame at 100 MHz
CLOCK_MHZ=100
WWIDTH=96                                          # --mvau-wwidth-max; 72 is INFEASIBLE at 512x512 / 250 fps (4x the pixels of the 256 net), 96 (16 x 6-bit weights per PE) is the smallest cap that works and gives the same design as no cap; the FINN auto-fold control uses 96 too
