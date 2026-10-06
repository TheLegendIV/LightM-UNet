# S12_dense_256_ratchet_ablation_v1

Ratchet ablation for the 256x256 S12 dense **nearest-upsample (no conv after the upsample), ReLU** net (`config_12_dense_relu_nearest_upsample_256`), the 256 counterpart of
`S12_dense_dsr_ablation_v1`. Uniform INT6, 250 fps (every node <= 400,000 cycles per frame at 100 MHz), `--mvau-wwidth-max 72`, `--force-dsp`, objective `--min-resources`,
FIFO / DWC modelling on (`--model-fifos`), no latency cap (the MILP's sum of node cycles is not a latency).

## Arms

| arm | MILP flags (on top of the common ones in `run_ablation.sh`) | |
|---|---|---|
| `ratchet_1pct` | `--ratchet-pct 1 --ratchet-floor 0.33` | |
| `ratchet_25pct` | `--ratchet-pct 25 --ratchet-floor 0.33` | |
| `ratchet_100pct` | `--ratchet-pct 100 --ratchet-floor 0.33` | |
| `ratchet_200pct` | `--ratchet-pct 200 --ratchet-floor 0.33` | |
| `ratchet_off` | `--ratchet-pct none` | no rate rule (element-rate DSR is off by default) |
| `analytical_25pct` | `MILP/analytical/net_fold.py --ratchet-pct 25 --ratchet-floor 0.33 --mvau-wwidth-max 72` (`run_analytical_arm.sh`) | **analytical flow**, not the MILP: same net / bits / fps / width cap / floor |
| `analytical_25pct_finnfifo` | the SAME folding as `analytical_25pct`, folding json written WITHOUT the FIFO lists (by `summarize_arms.py`) | **hardware-only variant**: the bridge forces nothing, FINN's own rtlsim autosizer (`largefifo_rtlsim`) sets every FIFO depth |
| `ratchet_ablation_finn_autofold` | none: FINN's own `step_target_fps_parallelization` (250 fps, `mvau_wwidth_max` 72) | **built on hardware**, not a MILP arm |

(The first sweep used 1/2/4/8/16 %: 1, 2, 4 and 8 % gave the identical folding and 16 % was a second one, so the sweep was widened to 1/25/(50)/100/200 %; the 50 % arm gave the same folding as 25 % and was dropped.)

**Ratchet** = "downstream no slower than upstream" in cycles per frame, on every compute node (conv / MVAU layers, the downsampling pad-MVAU, the argmax) against its nearest compute ancestor
(thresholds, dup, add, pools, concat, upsample and the input quantizer are relayed through): `cycles[C] <= max(floor * F, (1 + pct/100) * cycles[P])`, F = 400,000.

**Why floor 0.33 and not 0.** The floor is the number of cycles a node may always use whatever its upstream node does (0.33 * F = 132,000). With floor 0 the ratchet is infeasible below +50 % (tested at 1, 5, 12, 25, 33 %: infeasible;
50 % and up: feasible, with or without the width cap): the last blocks have tiny layers (`regular5`: 1 -> 4 channels, at most 65,536 cycles of work each) in front of the final transposed conv, which has 4x the work per pixel, so no
strict "no slower" chain can hold without a floor. From about 0.4 up (160k cycles and more) the design (~147k cycles) never touches the ratchet and the small-pct arms collapse to one folding; 0.33 is the smallest round floor
that is feasible for all arms and still binds. (The first version of the ratchet compared every node, thresholds and dups included, and was infeasible at any floor below 0.328; it now only ratchets compute nodes.)

**Why `--mvau-wwidth-max 72`.** It is FINN's own `mvau_wwidth_max` (SetFolding's cap on weight bits x SIMD per MVAU: 6-bit x SIMD 12). The MILP carries it as a hard cap only so that the MILP arms and the FINN auto-fold control
live in the same search space (the auto-fold control must be given the identical value). It is not needed for correctness, and on this net it hardly matters: with floor 0.33 the ratchet arms are identical at cap 72 / 96 / 144 / none;
only `ratchet_off` moves (101 DSP at 72, 105 DSP and 94.5k LUT with a looser cap).

## Results (`arms_summary.csv`, written by `summarize_arms.py`)

| arm | LUT incl. FIFO+DWC | LUT nodes | BRAM18 | DSP | slowest node (cycles) | fps | layers differing from `ratchet_off` (of 88) | folding |
|---|---|---|---|---|---|---|---|---|
| ratchet_1pct | 99,161 | 96,255 | 101 | 178 | 147,971 (`initial.conv`) | 675.8 | 82 | A |
| ratchet_25pct | 98,466 | 95,625 | 101 | 147 | 147,980 (`down1.conv.0`) | 675.8 | 82 | B |
| ratchet_100pct | 95,913 | 93,909 | 101 | 115 | 294,912 | 339.1 | 2 | C |
| ratchet_200pct | 95,558 | 93,629 | 101 | 113 | 327,680 (`final.argmax`) | 305.2 | 1 | D |
| ratchet_off | 95,274 | 93,387 | 101 | 101 | 327,680 (`final.argmax`) | 305.2 | 0 | E |
| analytical_25pct | 87,995 | 76,918 | 205 | 241 | 147,715 (`regular5.0.conv`) | 677.0 | 82 | F |

* **Six distinct foldings** (A-F; F = analytical) and seven builds: `analytical_25pct_finnfifo` shares folding F with `analytical_25pct` but is its own build, so `analytical_25pct` (FIFO depths forced from the analytical simulation)
  against `analytical_25pct_finnfifo` (FINN's autosized FIFOs) isolates the effect of the FIFO strategy at identical folding (compare BRAM, LUT and rtlsim cycles); the MILP / analytical FIFO columns of the table are blank for it, one per arm except that the dropped 50 % arm (run earlier) gave the same folding as 25 %. One hardware build per distinct folding (`arms_to_build.txt`, `arm_build_map.csv`).
* The knee is between 25 % and 100 %: up to +25-50 % the design keeps the whole network at ~147k cycles (676 fps); at +100 % the convs may sit at 294,912 cycles (339 fps, 2 layers differ from `off`); at +200 % only the argmax (PE 1,
  327,680 cycles) differs from `off`. Tight ratchets cost up to ~4% LUT and 77 DSP against `off` and buy 2.2x throughput.
* Every ratchet arm passes the edge check in `summarize_arms.py` (0 violations), every layer is INT6, `mvau_wwidth_max` 72 holds.
* **Analytical arm** (F): 677 fps, 241 DSP, 76.9k node LUT and 205 BRAM18 (143 of them simulated skip / prefetch FIFOs, FINN will delete the depth-2 ones). It is the same design the analytical artifact
  `S12_dense_256_u4_analytical_v1` already had (the width cap and the floor 0.33 / 25 % do not change it: block budgets 184,320 and 163,840 cycles, nodes at 147k, max SIMD x bits = 48). Against the MILP arms it uses
  more DSP (241 vs 113-178 for the fast arms) and fewer LUT nodes (76.9k vs 93.6-96.3k): different tie breaking at equal cycles (it takes the cheapest BRAM / LUT fold, the MILP's objective weighs DSP heavily) and the MILP prices
  `residual_add` separately (7.4k LUT) where the analytical flow merges it into `out_act`. Its BRAM18 is the simulated FIFO memory (the MILP's 101 BRAM18 is its FIFO model: skip + prefetch only).
  The analytical ratchet is block to block (a block's budget = 1.25 x the slowest node of the previous block), the MILP's is node to node, so the analytical design breaks the MILP rule once
  (`regular5.0.reduce.0` 65,536 -> `regular5.0.conv` 147,715 cycles; `ratchet_violations` = 1 in the table). At 4 % + floor 0.33 the analytical flow does not work: the block budgets ratchet down to ~136k cycles and
  the up5 block fails its simulation check (9.32 cyc/px steady against an 8.3 target), hence 25 %.
* FIFO model (BRAM18 101 in every arm) counts only skip FIFOs, prefetch FIFOs and DWCs; the down-block `Dup -> SWG_r` feed and the up / initial `FIFO main` are not priced (no closed form).

## Files

* `run_ablation.sh` (run in `lightmunet_dev`, all MILP arms in parallel, ~40 s), `run_analytical_arm.sh` (~5 min: the blocks are simulated in parallel workers), `summarize_arms.py`.
* Per arm folder: `layer_bits_folding_<arm>.json` (per_layer, extra_nodes, `intra_block_fifos` = the priced skip / prefetch FIFOs in the analytical role vocabulary the FINN bridge reads, `inter_block_fifos` (depth 2), `dwcs`,
  `_diagnostics.ratchet`, `_diagnostics.fifo_model`), `layer_bits_SITES_<arm>.json` (uniform INT6 per-site bits), `run_args.json`, `solve.log`, `summary.csv`, `final_output.onnx`.
* `arms_summary.csv`, `arm_groups.json`, `arm_build_map.csv`, `arms_to_build.txt`.

## Hardware

`hardware/builds/S12_dense_256_ratchet_ablation_v1/` (see its README): `STEP=bridge bash run_arms.sh` first (preamble + bridge dry run + landed-folding gate, no Vivado), then `bash run_arms.sh` for the
partition-2 OOC builds of the five distinct foldings plus the FINN auto-fold control (target 250 fps, `mvau_wwidth_max` 72), then `collect_results_partitions.py --arm-map arm_build_map.csv`.

## Caveats

* Reproduce with `bash MILP/artifacts/S12_dense_256_ratchet_ablation_v1/run_ablation.sh` after any change to `finn_milp.py`; the ratchet / FIFO model are new (2026-10-06).
* FINN still runs its rtlsim FIFO autosizer before the MILP depths are forced (`step_force_fifo_depths_from_milp`): FIFOs that the MILP does not list keep FINN's autosized depth, which inflates the BRAM of the hardware
  numbers relative to the MILP's FIFO model. The bridge reports matched / unresolved FIFOs (`fifo_force_report_partition_<i>.json`).
* The argmax PE and the FMPadding_Pixel SIMD are not applied by the bridge (only matters for partitions 5-7).
