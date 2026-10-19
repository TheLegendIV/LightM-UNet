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

## FINN-native build: `finn_auto/cap40_fps150/uniform_int6`

Same uniform INT6 network as `cap40_fps150/uniform_int6`, but with **FINN's own auto-folding** (`step_target_fps_parallelization`, 150 FPS target at 100 MHz, no MILP folding, no MILP FIFO depths, FINN's stock MaxPool lowering) and **FINN's rtlsim-based FIFO autosizing** (`--fifo-autosize rtlsim`). Job tag `S12_dense_512_bilinear_int6_cap40_finnauto_v1`; launcher `hardware/builds/S12_dense_512_bilinear_uniform_jobs/_launch_finn_auto.sh`.

Partitions 0-6 built normally. **Partition 7 could not be placed**: FINN's FIFO autosizing gave it FIFOs of 1,310,718 x 16 bit, 262,144 x 6 bit and 262,144 x 80 bit words (about 46 Mbit in total) next to PE=1 / SIMD=1 nodes, so Vivado stops in `place_design` with DRC `UTLZ-1` (1,300 RAMB36 needed, 312 on the xczu7ev; even the largest installed part, xczu19eg, has only 984). Partition 7's numbers are therefore from the post-synthesis / post-opt checkpoint (`finish_p7.py`, `util_from_dcp.tcl`, `report/p7_synth_only/`): utilization only, no WNS / Fmax. Its single-image rtlsim does not complete (261,892 of 262,144 output beats after 7.6 M cycles, flagged as deadlock).

| Partition | LUT | FF | BRAM18 | BRAM36 | Fmax MHz | rtlsim cycles (1 image) |
|---|---|---|---|---|---|---|
| 0 | 4,411 | 4,177 | 4 | 178 | 241.0 | 854,040 |
| 1 | 20,962 | 20,641 | 77 | 81 | 251.2 | 628,947 |
| 2 | 24,331 | 17,492 | 88 | 123 | 162.3 | 889,666 |
| 3 | 21,086 | 14,561 | 72 | 122 | 195.7 | 868,379 |
| 4 | 24,437 | 17,718 | 92 | 124 | 188.2 | 906,313 |
| 5 | 21,983 | 15,093 | 65 | 118 | 197.0 | 818,875 |
| 6 | 11,355 | 12,023 | 34 | 120 | 238.6 | 604,889 |
| 7 (synth only) | 31,636 | 12,797 | 1 | 1,300 | - | incomplete (261,892/262,144 beats) |
| **Sum** | **160,201** | **114,502** | **433** | **2,166** | **162.3 (partitions 0-6)** | |

Compared with the MILP-folded INT6 build of the same network and target (96.7k LUT, 45 BRAM18 + 80 BRAM36 = 205 BRAM18eq, Fmax 190.2 MHz, all partitions place and pass rtlsim): the FINN-native design needs 160.2k LUT and **4,765 BRAM18eq (7.6x the device's 624)**, so it cannot be implemented on the ZCU7EV. The BRAM comes almost entirely from FINN's autosized FIFOs (about 120 BRAM36 per partition in 0-6 and 1,300 in partition 7).

Files: `report/` (per-partition `ooc_synth_*`, `rtlsim_*`, `ooc_synth_and_timing_per_partition.json` with the all-8 aggregate), `auto_folding_config.json`, `final_hw_config.json`, `preamble.log`, `build.log`, `finish_p7.log`.

## Inter-block FIFO `regular5.0.out_act -> final` (partition 7)

The FIFO simulation grows this FIFO to 512 words in the INT4 and INT6 foldings (needed to reach the 666,666-cycle target). The build bridge could not match it (the real FIFO sits behind a width converter whose consumer has no mapped role), so the `8way/` builds of INT4 and INT6 kept it at depth 2. `finn_s12_build_steps.py` now matches such a FIFO by its producer role. `p7fix/` holds partition 7 rebuilt with the fix:

| Arm | Partition 7 rtlsim cycles (8way -> p7fix) | LUT | BRAM18 | Fmax MHz |
|---|---|---|---|---|
| INT4 | 794,216 -> 592,001 | 3,432 -> 3,531 | 0 -> 0 | 194.1 -> 195.6 |
| INT6 | 783,526 -> 592,021 | 4,102 -> 4,232 | 1 -> 1 | 190.2 -> 193.4 |

The combined 8-way numbers above are from the original builds (depth 2); the fixed partition 7 has not been re-assembled into a combined design. The mixed build's folding keeps the FIFO at depth 2 (the 0.4 caps rejected the grown FIFO); INT8 relaxed is not affected (its simulation meets its target at depth 2).
