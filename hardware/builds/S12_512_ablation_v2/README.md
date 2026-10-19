# S12 512 bilinear precision ablation (v2): FINN build results

Build outputs of the ablation arms solved in [`MILP/artifacts/S12_512_ablation_v2/`](../../../MILP/artifacts/S12_512_ablation_v2). The folder layout mirrors the MILP one (`<solve>/<arm>`), so every arm here sits next to the solve and FIFO-simulated folding it was built from (`…/<arm>/simfifo/layer_bits_folding_*_simfifo.json`).

Only reports, configs and logs are kept (no `intermediate_models`, no IP / bitstreams). The scripts and the exported networks of each job are in `hardware/builds/S12_dense_512_bilinear_<arm>_<v>/` (and `…_uniform_jobs/` for the generic launchers).

## Layout

```
<solve>/<arm>/
  8way/        the original full 8-partition build: report/ (ooc_synth_*, rtlsim_*, estimate_*), hawq_folding_config_partition*.json (bridged folding),
               fifo_plan_partition*.json (FIFO depths the bridge wanted), time_per_step.json
  p7fix/       (INT4, INT6 only) partition 7 rebuilt alone (--probe) after the inter-block FIFO fix, see below
  preamble.log, build.log
```

| Folder | Job tag | MILP target | Caps | Notes |
|---|---|---|---|---|
| `cap40_fps150/uniform_int4` | `S12_dense_512_bilinear_int4_cap40_v1` | 150 FPS | 0.4 | min-resources |
| `cap40_fps150/uniform_int6` | `S12_dense_512_bilinear_int6_cap40_v1` | 150 FPS | 0.4 | min-resources |
| `cap40_fps150/mixed_468` | `S12_dense_512_bilinear_mixed468_cap40_v1` | 150 FPS | 0.4 | max accuracy, bits 4/6/8 |
| `relaxed/uniform_int8` | `S12_dense_512_bilinear_int8_relaxed_v1` | **100 FPS** | 1.0 | min-resources; not the 150 FPS target |

## Results (OOC synthesis of the 8 partitions, 100 MHz target; DSP column reads 0 in OOC reports)

| Arm | LUT | FF | BRAM18 + BRAM36 | Worst-partition Fmax (MHz) | FINN est. FPS @100 MHz | Combined rtlsim (cycles, 1 image) |
|---|---|---|---|---|---|---|
| INT4 | 79,788 | 70,240 | 37 + 32 | 194.1 | 169.2 | 2,278,947 |
| INT6 | 96,651 | 102,248 | 45 + 80 | 190.2 | 169.2 | 2,267,093 |
| Mixed 4/6/8 | 109,472 | 116,681 | 67 + 96 | 182.2 | 169.2 | 2,239,227 |
| INT8 relaxed (100 FPS) | 194,295 | 142,050 | 57 + 73 | 130.7 | 127.2 | 2,462,919 |

All partitions of all four builds pass the per-partition rtlsim without deadlock. The board PL clock is 187.48 MHz.

## Inter-block FIFO `regular5.0.out_act -> final` (partition 7)

The FIFO simulation grows this FIFO to 512 words in the INT4 and INT6 foldings (needed to reach the 666,666-cycle target). The build bridge could not match it (the real FIFO sits behind a width converter whose consumer has no mapped role), so the `8way/` builds of INT4 and INT6 kept it at depth 2. `finn_s12_build_steps.py` now matches such a FIFO by its producer role. `p7fix/` holds partition 7 rebuilt with the fix:

| Arm | Partition 7 rtlsim cycles (8way -> p7fix) | LUT | BRAM18 | Fmax MHz |
|---|---|---|---|---|
| INT4 | 794,216 -> 592,001 | 3,432 -> 3,531 | 0 -> 0 | 194.1 -> 195.6 |
| INT6 | 783,526 -> 592,021 | 4,102 -> 4,232 | 1 -> 1 | 190.2 -> 193.4 |

The combined 8-way numbers above are from the original builds (depth 2); the fixed partition 7 has not been re-assembled into a combined design. The mixed build's folding keeps the FIFO at depth 2 (the 0.4 caps rejected the grown FIFO); INT8 relaxed is not affected (its simulation meets its target at depth 2).
