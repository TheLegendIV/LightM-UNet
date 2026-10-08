#!/bin/bash
# The ONE definition of the target shared by every arm of this ablation (MILP arms, the analytical arm, the FINN auto-fold control on hardware). Sourced by run_ablation.sh and run_analytical_arm.sh;
# summarize_arms.py checks that every result file carries exactly these values.
CONFIG=config_12_dense_relu_nearest_upsample_256   # 256x256 S12 dense nearest-upsample (noconv) ReLU net
BITS=6                                             # uniform weight = activation bits
FPS=250                                            # every node <= clock / FPS = 400,000 cycles per frame at 100 MHz
CLOCK_MHZ=100
DSR_FLOOR=0.33                                     # --dsr-floor: smallest round floor that is feasible for every arm (see run_ablation.sh)
WWIDTH=72                                          # --mvau-wwidth-max; the FINN auto-fold control on hardware uses the same value (run_arms.sh)
