# S12 dense 512 bilinear, uniform INT8, max-FPS folding (`S12_dense_512_bilinear_int8_maxfps_v1`): final build results

Reports, configs and logs copied back from the FINN container (`finn_persistent`, `finn_deployment_outputs/`). No `intermediate_models` / IP.
This is the **v1** INT8 design, the one before the v2 re-fold (`S12_dense_512_bilinear_int8_v2`), and it has the highest measured FPS of all builds: 356.5 FPS steady state on the board, Dice 0.7943 (see `deployment/results.csv`; v2 reached 247.5 FPS, the original INT8 214.5 FPS).

## The final design is assembled from three builds

| Partitions | Folder here | Container output dir (`S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_…`) | Why |
|---|---|---|---|
| 0 | `ipool_p0_probe0_20261016_110642/` | `ipool_p0_milpfold_probe0_20261016_110642` | initial pool lowered with **InferPool** (`ipool_preamble`) |
| 1, 5, 7 | `p157fix_probe1_5_7_20261016_014355/` | `p157fix_milpfold_probe1_5_7_20261016_014355` | name-stable FIFO insertion (p1 deadlock / p5 mis-sized up4 FIFOs), LabelSelect PE = labels, final bias PE, final FMPadding_Pixel SIMD (p7) |
| 2, 3, 4, 6 | `8way_pairs_20261015_210334/` | `pairs_milpfold_8way_20261015_210334` | the full 8-way run (partitions 0, 1, 5, 7 of this run are superseded by the rows above) |

Same composition as `_launch_golden_int8.sh` (per-partition golden check of the assembled design). Folding: `layer_bits_folding_S12_512_bilin_int8_maxfps_pairs.json` (in `ILP/artifacts/S12_512_bilin_int8_maxfps/simfifo_pairs/`). `logs/` has the build, preamble and bridge logs of the three runs.

## Final per-partition results (OOC synthesis, 100 MHz target; DSP column reads 0 in OOC reports)

| Partition | Source | LUT | FF | BRAM18 | BRAM36 | Fmax MHz | rtlsim cycles (1 image) | deadlock |
|---|---|---|---|---|---|---|---|---|
| 0 | ipool_p0 | 2,768 | 3,667 | 4 | 0 | 314.9 | 328,268 | no |
| 1 | p157fix | 35,213 | 35,325 | 8 | 1 | 295.9 | 449,091 | no |
| 2 | 8way_pairs | 36,536 | 26,736 | 6 | 27 | 284.5 | 452,702 | no |
| 3 | 8way_pairs | 29,675 | 21,670 | 6 | 26 | 248.1 | 442,001 | no |
| 4 | 8way_pairs | 36,990 | 27,568 | 8 | 28 | 262.5 | 461,007 | no |
| 5 | p157fix | 25,999 | 21,554 | 7 | 24 | 230.8 | 416,632 | no |
| 6 | 8way_pairs | 16,385 | 18,541 | 5 | 0 | 294.5 | 302,321 | no |
| 7 | p157fix | 6,426 | 9,659 | 5 | 0 | 186.9 | 525,485 | no |
| **Sum / min** | | **189,992** | **164,720** | **49** | **106** | **186.9** | | |

Note: `8way_pairs_*/report/rtlsim_performance.json` and `estimate_network_performance.json` (191.5 FPS estimate) are the combined-design reports of the *original* 8-way run, i.e. with the old partitions 0, 1, 5, 7 (its combined rtlsim shows 0 output transactions: the old partition 7 problem the p157fix rebuild solved). The assembled design above was never run through the combined rtlsim; use the per-partition rtlsim numbers and the board measurement.
