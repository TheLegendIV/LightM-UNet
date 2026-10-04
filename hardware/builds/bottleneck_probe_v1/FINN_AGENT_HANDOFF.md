# Handoff: bottleneck probe builds (for the agent that runs on the FINN container)

You are taking over a prepared set of build files. Everything host-side is done and tested; the FINN-side script has
**never run** (no FINN container was available when it was written). Your job is to make it run, run it, and report.
Read `AGENTS.md` (repo root) first: FINN code must be run as `docker exec -e HOME=/tmp/home_dir <container> ...`, files are
`docker cp`'d flat into `/home/thelegendiv/finn/notebooks/enet/`, Vivado needs `source /tools/Xilinx/Vivado/2022.2/settings64.sh`.

> **Sections 13-15 (final deconv, Pool route, batch protocol: build + OOC for ALL probes, max 4 at a time, per-block change log) are the newest and override earlier wording.**

## 1. The goal (do not lose sight of it)
**Verify that our tightly coupled bottleneck does not deadlock in hardware and achieves the required throughput.**
"Tightly coupled" = our PE/SIMD per node, thresholds at the narrowest PE that keeps up, DWCs where widths differ, and
**our own small FIFO depths forced onto the design** (almost every edge 2-4 words; one deep skip FIFO; one elastic FIFO
in front of FMPadding). The primary evidence is the **stitched-IP rtlsim**: it must finish (no deadlock) and its stable
throughput must be within 2% of the target of 72 cycles per output pixel. OOC synthesis (LUT/BRAM/DSP, fmax) is secondary
(`--ooc`); do rtlsim first for all cases.

## 2. What is being built
15 independent cases = dilation {1,2,4,8,16} x uniform INT {4,6,8}. Each is **one regular ENet bottleneck as a single
partition** (no 8-way flow, no `finn_s12_build.py`): Cin = Cout = 32, Cmid = 8 (v = z = 4), dense 3x3 conv, stride 1,
padding = dilation, 32x32 map, ReLU, identity skip + residual add, weights random with `torch.manual_seed(0)`.
Graph after export: `Thr_in (QuantIdentity stand-in for the previous block) -> Dup -> reduce / conv / expand MVAUs with
thresholds -> Add -> Thr(residual_add) -> Thr(out_act)`; the merge pass collapses the last two thresholds into one `Thr_out`
(that is the graph the analytical model describes). Dilation 16 on 32x32 is a degenerate padding-dominated stress case
(window wider than the map, skip FIFO ~17.5k words); expect it to be the largest/slowest to simulate.

## 3. Files (all under `hardware/builds/bottleneck_probe_v1/` unless noted)
| File | Role | Status |
|---|---|---|
| `export_bottleneck_probe.py` | Brevitas -> QONNX per case (runs in `lightmunet_dev`, torch+brevitas+qonnx) | done, ran, 15 ONNX written |
| `inputs/<case>.onnx`, `_probe.json`, `_folding.json` | model, case parameters, role-keyed folding + FIFO config + model prediction | done (45 files) |
| `make_folding_configs.py` | regenerates `_folding.json` from `MILP/analytical/bottleneck.py` (+ `bottleneck_sim.py` verification) | done |
| `../../finn_compose_thresholds.py` | `ComposeConsecutiveMultiThresholds` + `step_compose_consecutive_thresholds` | done, 11 unit tests (`../../checks/test_compose_thresholds.py`) exact-match |
| `finn_bottleneck_probe_build.py` | the FINN build of one case (container) | **UNTESTED, yours to fix** |
| `run_probes.sh`, `collect_probe_outputs.sh` | host-side launcher (docker cp + parallel exec) and result collector | syntax-checked only |
| `compare_probe_vs_model.py` | verdict + predicted-vs-landed table from collected `probe_result.json` | compiles, untested on real data |

`MILP/analytical/` has the model (`bottleneck.py`), the cycle-level simulator (`bottleneck_sim.py`) and `analytical.md`.
`bottleneck.py ... --folding-json PATH` regenerates a case's config; `--onnx` writes a Netron graph of the verified design
(example: `MILP/analytical/outputs/bottleneck_ref_cin32_d8_T72_int4.onnx`).

## 4. Do these in order (cheap gates first; do not launch the batch before gate 3 passes)
1. **Container**: `docker ps` for the FINN container name (changes across recreations). Check FINN version and that
   `FMPadding_*`, `ConvolutionInputGenerator_rtl`, `MVAU_rtl`, `Thresholding_rtl`, `StreamingFIFO_*`, `StreamingDataWidthConverter_*`
   op names match what the script assumes (it matches by `op_type.startswith(...)`, so FMPadding_hls/rtl both work).
2. **Merge-pass test in the container** (it must also pass with FINN's qonnx): `python3 hardware/checks/test_compose_thresholds.py`.
3. **Dry run of one case up to folding**: `CONTAINER=<name> CASES=bottleneck_cin32_d8_int4 STOP_AFTER=folding bash run_probes.sh`,
   then read `/tmp/probe_bottleneck_cin32_d8_int4.log` and `<out>/probe_result.json`. Verify:
   - every step ran; `after_convert_to_hw.onnx` contains `Thresholding` (standalone, `noActivation=1` MVAUs), one `DuplicateStreams`,
     one `AddStreams`, one `FMPadding`, one `ConvolutionInputGenerator`, three `MVAU`; and **exactly one** Thresholding after the Add
     (if two remain, the merge pass did not match: print the MultiThreshold chain before `convert` and fix the pass, see 6.a);
   - `identify_roles` resolved all roles (thr_s = the Dup output that does not lead to the reduce MVAU; thr_e = consumer of expand);
   - landed PE/SIMD in `stages.folding` equal `predicted.nodes` in the `_folding.json` (the compare script flags mismatches).
4. **Run to FIFO**: `STOP_AFTER=fifo`. Inspect `stages.fifo`: each forced edge should list `stock_depth` and `forced_depth`; edges
   reported as "not in prediction" need a look (FINN may have inserted a FIFO or DWC our simulation did not, or removed shallow FIFOs).
5. **Run the full flow for d=8 INT4 to rtlsim** (default `STOP_AFTER=rtlsim`). Success = `stages.rtlsim.steady_cyc_per_pixel` ~ 72
   (<= 73.5) and the run terminates. Then do the **negative control**: `EXTRA="--skip-scale 0.5"`: with half the skip FIFO the design
   is expected to deadlock/stall (rtlsim timeout, exit 124) -- that proves the test can fail. Also run `EXTRA="--fifo-policy stock"` as a control.
6. **Batch**: all 15 (`JOBS=4`), then `collect_probe_outputs.sh`, then `compare_probe_vs_model.py`. Then repeat with `EXTRA="--ooc"` (or add
   `--ooc` in the first batch if Vivado time is acceptable) and copy `utilization_placed.rpt` out of `/tmp/finn_dev_*` before the container goes away.

Useful existing tools for gates 3-4 (found by the exploration pass, not verified against this FINN version):
`hardware/dump_milpfold_landed_partition.py` applies a folding JSON and re-runs `AnnotateCycles` in seconds without HLS/Vivado
(hardwired to the 8-way flow, so it needs generalising for one block; the probe script's own `identify_roles` + `stages.folding`
output covers the same check), and `hardware/utils/dump_fifo_depths.py` dumps landed `StreamingFIFO_rtl` depths and LUT/FF/SRL cost.

## 5. Where I expect trouble (untested assumptions in `finn_bottleneck_probe_build.py`)
a. **Merge pass precondition.** It matches `MultiThreshold -> [scalar Mul/Add]* -> MultiThreshold` with one consumer, same data layout, integer-valued
   stage-1 levels. In the real streamlined graph the pair may be separated by something else (a Transpose, a non-scalar Mul, a Dup if the
   forked-dequant steps moved it). Print the chain after `step_enet_streamline` + the three fork fixups and extend the matcher; keep the
   unit tests green (exact equality is the contract). Use `--no-merge` to build the unmerged graph as a control (it will then have two
   thresholds after the Add, unlike the model).
b. **Step list.** It mirrors `enet_ip_partitioned_8way_steps` up to `_fixup_degenerate_signed_bias`, plus merge, plus the live
   RTL-MVAU convert, then `step_create_dataflow_partition` (child model), `step_specialize_layers`. If the archived
   `hardware/archive/20261001_refactor/probes/finn_build_probe_s12_context_v2.py` has a step ordering that works better for single blocks
   (it has `step_fix_signed_thresholds`), borrow it. The input is `QuantIdentity(unsigned)`; if FINN complains "Signed output requires actval < 0"
   use that step.
c. **FIFO forcing** (`step_force_fifo_depths`). It matches `StreamingFIFO*` nodes to predicted edges by the roles at both ends (DWC = `dwc`).
   Open questions you must settle against the actual FINN version: does `step_set_fifo_depths` leave depth-2 FIFOs or remove "shallow" ones;
   is `depth` the only attribute to change, or must `impl_style` (rtl vs vivado) also follow the new depth (deep skip FIFO!); are DWCs inserted
   before or after the FIFO sizing so the edge list matches `cfgj["fifos"]`. After changing depths it re-runs `SplitLargeFIFOs`, `PrepareIP`,
   `HLSSynthIP` (same recipe as `hardware/finn_s12_build.py:127-139`). If an edge FINN has but we do not predict exists, keep stock depth (never below 2).
d. **rtlsim.** `step_measure_rtlsim_performance` is called with `rtlsim_batch_size = 3`; steady throughput comes from `stable_throughput[images/s]`
   in `report/rtlsim_performance.json` (key names may differ: the script falls back to `throughput[images/s]`; with a single frame the
   number includes pipeline fill and is not a steady rate, so make sure it really runs >= 2 frames). A deadlock presents as the rtlsim never finishing:
   `run_probes.sh` wraps each build in `timeout $TIMEOUT` (default 4 h); reduce it for the negative control. Frame cycles to expect for d=8 INT4: ~74k
   per frame at steady state (72 x 1024).
e. **Folding attributes.** Applied directly with `set_nodeattr` (`PE`, `SIMD`, `parallel_window`, `depth_trigger_bram`); `step_apply_folding_config`
   is not used. If `parallel_window` or `depth_trigger_bram` are not settable on some op, drop them (here `parallel_window = 0` everywhere,
   thresholds use `depth_trigger_bram = 1024` = "block").
f. **Env.** Container python needs `PATH` to include Vitis HLS / Vivado bins (the script sets them) and `FINN_BUILD_DIR` (per run, under the output dir).
   Concurrent builds must have separate build dirs; `run_probes.sh` runs builds in parallel because each gets its own `FINN_BUILD_DIR`.

## 6. Known repo issue (not yours to fix, just be aware)
`enet/nnunetv2/nets/LayerQuantENet.py` in the working tree is missing its `from nnunetv2.nets.QuantENet import (...)` / `ENet` pattern imports
(an uncommitted edit deleted them; `git diff` shows it). The export script injects `_quant_conv2d` / `_quant_block_act` so it still works. The
user should restore the import block before the next training/export run.

## 7. What "the answer" looks like -- report back with
For each case: `verdict` from `compare_probe_vs_model.py` (PASS / SLOW / NO-RTLSIM), rtlsim steady cyc/px vs 72 vs the simulator's 72.00, rtlsim
first-out latency vs the model's (d=8 INT4: ~19.8k cycles), the forced vs stock FIFO depths, any edges where FINN disagreed with the prediction,
and -- if `--ooc` ran -- LUT/BRAM18/DSP/fmax vs `predicted.totals` (model: d=8 INT4 = 2451 LUT, 3 BRAM18, 16 DSP; INT8 moves several thresholdings to
BRAM, expect ~34 BRAM18). Append rows to `hardware/results.csv` via `hardware/collect_results.py` with `model_name=bottleneck_cin32_d{d}_int{b}`,
`config=T72_merged` (the OOC `DSP` field is wrong; use `--dsp-rpt-dir`). Any deadlock is the headline finding: capture which FIFO was full
(`dump_fifo_depths.py` / rtlsim wave) and compare with `MILP/analytical/bottleneck_sim.py` (it can reproduce a deadlock by lowering `skip_depth`, see
`test_undersized_skip_fifo_deadlocks` in `MILP/analytical/test_bottleneck_sim.py`).

## 8. Reference facts from the model (for sanity checks)
- Reference case d=8 INT4: MVAU_r PE1xSIMD4 (64 cyc/px), MVAU_m PE1xSIMD8 (72), MVAU_e PE1xSIMD4 (64); all thresholds PE 1; SWG/FMPad SIMD 8
  (not parallel_window); DWC LUTs ~62 total; skip FIFO 8736 words x 4 bit; FIFO feeding FMPad 14 words (needs >= pad+1 = 9); FIFO feeding MVAU_r 4 words
  (2 loses throughput); every other FIFO 2 words.
- Why those FIFOs: at each output row start the SWG needs pad+1 new real pixels at once (elastic FIFO), MVAU_r reads its input in a burst (feed FIFO),
  and the skip branch must buffer what arrives during the SWG fill (about half the image at d=16). The simulator finds that with depth-2 FIFOs everywhere the
  block runs at ~80 cyc/px, not 72. If hardware shows a similar loss, check those two FIFOs first.
- Depth 1 would be a combinational ready path; real FINN FIFOs are >= 2 (skid buffer), so do not try to go below 2.


## 9. Downsampling probes (6 more cases) -- added after the regular-bottleneck batch

Goal is the same (no deadlock in hardware, throughput within 2% of the target), now for the ENet **downsampling** bottleneck:
`dn_cin16_cout32_in64_int{4,6,8}_{fmpad,mvau}`. One block = one partition = one single-partition run, exactly like section 4
(use `SET=dn CONTAINER=<name> bash run_probes.sh`; `STOP_AFTER=folding` first). Model: `MILP/analytical/dn_bottleneck.py`
(+ `dn_bottleneck_sim.py`, `test_dn_bottleneck.py`, section "Downsampling bottleneck" in `analytical.md`).

Case: Cin 16 -> Cout 32, internal ratio 4 (Cmid 8), 64x64 input -> 32x32 output, uniform INT b, frame budget F = 72*32*32 = 73728
cycles (T_out = 72 cyc per OUTPUT pixel, T_in = 18 per input pixel). The target for rtlsim is therefore 72 cyc per output pixel
(`predicted.T` in the folding json; compare script already uses it). Dummy weights, seed 0. No dilation (3x3 d = 1, pad 1).

Graph (after merge pass): `Thr_in -> Dup -> [main] SWG_r(2x2 s2) -> MVAU_r -> Thr_r -> FMPad(3x3) -> SWG_m -> MVAU_m -> Thr_m -> MVAU_e -> Thr_e -> Add`,
`[skip] Dup -> MaxPool -> (A) identity MVAU_s -> Thr_s | (B) channel pad FMPad_c -> Thr_s -> Add`, `Add -> Thr_out`.

* **`mvau` variants (3 probes) are the comparison baseline** = the existing `FINNDownsamplingBottleneck` unchanged: skip = MaxPool ->
  frozen padded-identity 1x1 conv (INT8 weights) -> MVAU_s. They need NO new FINN pass: do these three first (`CASES="dn_..._mvau"`).
  Folding roles: dup, swg_r, mvau_r, thr_r, fmpad, swg_m, mvau_m, thr_m, mvau_e, thr_e, maxpool, mvau_s, thr_s, add, thr_out
  (`identify_roles_down` identifies MVAUs by (MW, MH) = (4*Cin, Cmid), (9*Cmid, Cmid), (Cmid, Cout), (Cin, Cout); the two sliding windows by
  ConvKernelSize 2 vs 3; the MaxPool, Dup, Add by op type).
* **`fmpad` variants (3 probes) replace the identity MVAU by a channel zero-pad**: the exported ONNX has `MaxPool -> Pad(pads [0,0,0,0, 0,16,0,0],
  value 0) -> Quant` (the residual add's shared input quantizer, i.e. the skip threshold sits AFTER the pad = "pad_thr" order in the model;
  the model's default "thr_pad" order would need a different export, not done). FINN has no hardware op for a channel pad, so a **new pass
  `hardware/finn_channel_pad.py::step_channel_pad_to_fmpadding` must be written by you** (currently a stub raising NotImplementedError; the
  full contract is in its docstring). Idea: the stream is NHWC, so re-read it as `[N, H*W, C/s, s]` (s = 16 here) and pad the "width" axis
  (C/s) on the right by (Cout-Cin)/s = 1 with FMPadding: ImgDim [1024, 1], NumChannels = SIMD = 16, Padding [0,0,0,1] (the exact values are
  in `folding["fmpad_c"]`). The graph keeps its real tensor shapes. Test it on the tiny example in `MILP/analytical/analytical.md` (2x2, C 2 -> 6,
  s = 2) before trusting it, in whole-graph rtlsim. If FINN rejects this, report why: the alternatives are a StreamingConcat with a
  zero stream (needs a zero source) or going back to the identity MVAU.
* **Differences from the regular block to watch for in the logs:** two pixel domains (Dup, MaxPool, SWG_r see 64x64, the rest 32x32);
  MaxPool is not foldable (one pixel = all Cin channels per cycle, ~1.25*64*64 cycles per frame) so a width converter widens Dup's stream to
  Cin*A bits in front of it; the strided 2x2 sliding window has a ~2W-pixel buffer (no elastic FIFO was needed in the model); the only deep
  FIFO is the skip FIFO (`skip FIFO`, 1662 words x A bit for fmpad/pad_thr, 1259 for mvau at INT4), every other edge is 2-4 words.
* **Expected numbers from the model (INT4 / INT6 / INT8):** fmpad LUT 3208 / 3885 / 4511, BRAM18 5 / 10 / 38, DSP 20 / 16 / 16;
  mvau LUT 3520 / 4265 / 4898, BRAM18 5 / 11 / 42, DSP 28 / 24 / 24; steady 72.00 cyc per output pixel; first-out latency about 3.8k (fmpad)
  / 4.1k (mvau) cycles. FMPadding LUT is not calibrated in the cost model (priced 0), so landed LUT of the fmpad variant is expected to
  exceed the prediction by the FMPadding + width converters.
* Negative control as in section 4: `EXTRA="--skip-scale 0.5"` should deadlock/stall the down block too (the model deadlocks below ~600 words
  of skip FIFO at INT4/mvau-order settings). Keep ordinary FIFOs at their forced 2-4 words for that test.


## 10. Follow-up to FINN_AGENT_PRELIM_REPORT.md -- READ THIS BEFORE RUNNING MORE BUILDS

Thank you for the report. Three things change the plan.

**10.1 The PASS verdicts were false positives (FINN's "stable throughput" is mis-scaled).** From your own numbers
(`N = 6`, `cycles` = 6-frame total, `latency_cycles` = the single-frame run): FINN's `stable_throughput` equals
`N / (cycles - latency_cycles)`, but `cycles - latency_cycles` covers only the other **N-1** frames. Correct steady period:

    cyc/px = (cycles - latency_cycles) / (N - 1) / pixels_per_frame        (pixels_per_frame = 1024 for the regular blocks)

| case | FINN field | corrected | target |
|---|---|---|---|
| d1_int4 | 61.39 | **73.67** | 72 |
| d2_int4 | 63.17 | **75.80** | 72 |
| d8_int4 (3 frames) | 58.61 | **87.92** (= (cycles-lat)/2) | 72 |

`steady_cyc_per_pixel` in `finn_bottleneck_probe_build.py` must use `(N-1)`; `compare_probe_vs_model.py` already recomputes it from the
raw `cycles` / `latency_cycles` / `N` fields (and prints FINN's number only for reference). Verdicts: PASS <= 2%, NEAR <= 3%, SLOW otherwise.
Single-frame totals (`latency_cycles`) match the model to 0.3% (76115 vs 76079, 78255 vs 78260, 91130 vs 91325): the per-frame model was right,
**consecutive frames just do not overlap as the single-frame simulation assumed.**

**10.2 Cause and cure.** The sliding window (and the FMPadding feeding it) serves ONE frame at a time. Frame k+1's first window needs
`n_fill = pad*W + pad + 1` freshly computed real pixels (d1: 34, d2: 67, d4: 133, d8: 265, d16: 529 at cf = 1) and the upstream (MVAU_r, ~64 cyc/px)
delivers them only after frame k's last window is out, so every frame pays about n_fill x 64 cycles of gap (d8: ~17k of 74k). The simulator
(`bottleneck_sim.simulate(..., frames=3)`) now models this and reproduces the hardware to 0.6% (73.85 / 76.03 / 88.31 for d1 / d2 / d8).
The cure is elasticity in front of the window, not a faster node: the FIFO feeding FMPad must hold the next frame's first window
(`>= n_fill*cf + 2` words) and the FIFO after FMPad the next frame's top padding rows (`>= pad*(W+2*pad)*cf + 2`). With those the simulated
period is 72.0 / 72.2 / 72.3 / 72.7 for d1 / d2 / d4 / d8 and 74.0 for d16 (a residual 2.7% that no FIFO removes: the window reads the next
frame's padding rows at one word per cycle before it can emit; flagged as a warning). **The `_folding.json` files in `inputs/` have been
regenerated with these depths (edge `fmpad -> swg` is new in the FIFO list). Re-run all regular cases with them; the old runs used the
old depths and are expected to be SLOW.** Memory note: the prefetch FIFOs are `32 bit x ~270` words at d8 (about 1 BRAM18 or ~300 LUT as SRL),
cheap compared to the 22% throughput they recover.

**10.3 FIFO implementation changes the resource prediction (open question for you).** You found that overriding a FIFO depth forces
`impl_style="rtl"` (SRL) because the Vivado FIFO IP only accepts a power-of-two depth menu (16 ... 32768) and rtlsim supports only rtl. So in the
probes the 8736-deep skip FIFO is built from SRL LUTs (about `width * ceil(depth/32)` LUT, ~1.1k LUT at INT4), whereas the model priced it as BRAM
(3 BRAM18, unrounded). For the real design we want BRAM (BRAM-bound study, narrow+deep FIFO): that needs `impl_style="vivado"` with the depth
rounded UP to the next power of two (8736 -> 16384, 4 bit wide -> 4 BRAM18; 1662 -> 2048 ...). Please report both numbers if you can: (a) OOC of the
rtl/SRL build as is, (b) one OOC run with the skip FIFO forced to `impl_style="vivado"` at the rounded depth (rtlsim is not available for it:
verify throughput with the rtl build, then swap the FIFO only for the synthesis). The model will get a `skip_fifo_impl` knob once we know
(b) works.

**10.4 Housekeeping.** Eight parallel builds were fine. Keep `--rtlsim-frames 6`. `collect_probe_outputs.sh` also collects `dn_cin16_*` dirs;
stale dirs from the iteration phase can be ignored (the compare script lists every dir it finds, so delete or move them before comparing).
The build script on the host was extended with downsampling support (section 9): `identify_roles_down`, a `custom` flag in the folding
dict (skipped by the generic apply), and the `finn_channel_pad` hook -- your container copy does not have these edits yet; `run_probes.sh`
re-copies the host files, so just rerun it (check that your fixes 1-5 are still in the host copy, they were when I extended it).


## 11. Upsampling probes (6 more cases): `up_cin32_cout16_in32_int{4,6,8}_{conv,noconv}`

Same goal and procedure as sections 4 / 9 (`SET=up CONTAINER=<name> bash run_probes.sh`; `STOP_AFTER=folding` first). Model:
`MILP/analytical/up_bottleneck.py` (+ `up_bottleneck_sim.py`, `test_up_bottleneck.py`, section "Upsampling bottleneck" in `analytical.md`).
Measure steady throughput as in section 10.1: `(cycles - latency_cycles) / (N - 1) / 4096` -- the target is **18 cyc per OUTPUT pixel**
(`predicted.T`; frame budget F = 73728, same as the 72-cycle 32x32 probes; input side 72 cyc per input pixel).

Case: ENet up4-like, nearest-neighbour decoder: Cin 32 -> Cout 16, Cmid = Cin // 4 = 8, 32x32 input -> 64x64 output, uniform INT b, seed 0.
* `conv` = decoder_type `nearest_conv_upsample` (main = main_proj 1x1 -> nearest x2 -> 3x3 skip_resize_conv; the decoder this repo trains);
  `noconv` = `nearest_upsample` (no 3x3). Ext branch for both: reduce 1x1 -> ConvTranspose 2x2 stride 2 -> expand 1x1, `Add`, `Thr_out`.
  Both branches are real compute here, unlike the bottleneck's identity skip.
* Expected FINN graph (after the usual convert): `Thr_in -> Dup`; main `MVAU_p -> Thr_p -> UpsampleNearestNeighbour -> [FMPadding 3x3 -> SWG 3x3 -> MVAU_k -> Thr_k]`;
  ext `MVAU_r -> Thr_r -> FMPadding_Pixel -> SWG 2x2 -> MVAU_u -> Thr_u -> MVAU_e -> Thr_e`; `AddStreams -> Thr_out`. The ConvTranspose is lowered by the
  repo's `InferPixelPaddingDeconv` (FMPadding_Pixel + Im2Col + MatMul). `identify_roles_up` tells `mvau_r` (Cin -> Cmid) from `mvau_u` (4*Cmid -> Cmid, same
  (MW, MH) = (32, 8) here) by their producer (Dup side vs a 2x2 sliding window); `thr_s` (noconv) is the consumer of the UpsampleNearestNeighbour node.
* **Model facts to check against hardware:** MVAU_k (conv variant) is the heaviest node: PE 8 x SIMD 16 = 128 MACs per cycle, 64 DSP at INT4 (block total 100 DSP,
  vs 36 for `noconv`); MVAU_u does `4*Cmid*Cmid` MACs per output pixel because 3 of 4 window elements are inserted zeros; UpsampleNearestNeighbour is not
  foldable (one OUTPUT pixel per cycle = 4096 cycles per frame, ~6% of F); only the join FIFOs are deep: `skip FIFO` (end of the ext branch, ~1.2k words x 4 bit for
  `conv`) and `FIFO main` (17 words); prefetch FIFOs in front of the 3x3 FMPadding and FMPadding_Pixel are small (~100-140 words, `emap` in `verify_with_sim`).
  Expected steady period 18.1 (conv) / 17.6 (noconv): no deep padding window, so the inter-frame gap of section 10 is only ~0.6%. Expected first-out latency ~2.4k (conv) /
  ~0.14k (noconv) cycles.
* **Expected resources (INT4 / INT6 / INT8):** conv LUT 5598 / 7308 / 7971, BRAM18 3 / 3 / 25, DSP 100 / 100 / 100; noconv LUT 3146 / 4211 / 4894, BRAM18 1 / 1 / 21, DSP 36
  (FIFOs, DWCs and thresholds included; FMPadding_Pixel, FMPadding and UpsampleNearestNeighbour LUTs are not calibrated, priced 0).
* Folding roles: dup, mvau_p, thr_p, upnn, [fmpad_k, swg_k, mvau_k, thr_k | thr_s], mvau_r, thr_r, fmpadpix, swg_u, mvau_u, thr_u, mvau_e, thr_e, add, thr_out. No new FINN
  pass is needed (all ops exist), so these six can run right away; if `UpsampleNearestNeighbour` or `FMPadding_Pixel` rejects the forced `SIMD`, report the attribute error.

### 11.1 Three more upsampling probes at the ENet U4 up5 sizes: `up_cin16_cout4_in64_int{4,6,8}_conv` (`SET=up5`)

"U4" = ENet width divisor 4: (16, 64, 128, 64, 16) / 4 = channels (4, 16, 32, 16, 4); the up4 probes above are its 32 -> 16 block (32x32 -> 64x64), up5 is 16 -> 4
(64x64 -> 128x128, Cmid = 16 // 4 = 4), conv decoder (`nearest_conv_upsample`) only. Same frame budget F = 73728, so **T_out = 4.5 cyc per OUTPUT pixel**
(16384 output pixels; T_in = 18 per input pixel). Measure as in 10.1 with `pixels_per_frame = 16384` (`compare_probe_vs_model.py` does this for `block == "up"`).
* Everything of section 11 applies. New: with Cout = Cmid = 4 the 1x1 MVAUs `mvau_p` (16 -> 4) and `mvau_r` (16 -> 4) have the SAME (MW, MH); `identify_roles_up`
  now classifies the three 1x1 MVAUs by what follows them (UpsampleNearestNeighbour = proj, FMPadding_Pixel = reduce, AddStreams = expand).
* Model: every MVAU at 4.00 cyc/px (89% of the 4.5 budget), MVAU_k = PE 1 x SIMD 36 (36 DSP, the 3x3 on 4 channels), block 64 DSP / 3.7k LUT / 2 BRAM18 at INT4
  (INT6 / INT8: see `predicted.totals` in the folding json); simulated steady 4.38 cyc/px (3 frames), first-out latency ~1.2k cycles; skip FIFO (ext end) ~520 words x 4 bit,
  FIFO main 5 words, FIFO in front of the 3x3 FMPadding ~530 words. T_out is only 4.5 cycles: with so little slack per pixel, watch the stream widths (each DWC and the
  UpsampleNearestNeighbour node run at one word per cycle) -- a node that needs more than one cycle per output pixel anywhere on the join would show up as SLOW.


## 12. Initial-block probes (3 cases): `init_cin1_cout4_in256_int{4,6,8}`  (`SET=int`)

ENet U4 initial block, `FINNInitialBlockConcat` (LayerQuantEnetFINN.py): 1 -> 4 channels, 256x256 input -> 128x128 output. The block has its own input quantizer
(no stand-in). Model: `MILP/analytical/int_bottleneck.py` (+ `int_bottleneck_sim.py`, `test_int_bottleneck.py`, section in `analytical.md`).

    Thr_in -> Dup -> [FMPad -> SWG (3x3, stride 2) -> MVAU_c (9 -> 3) -> Thr_c] -> FIFO main -+
                     [MaxPool (2x2, stride 2) -> Thr_m] --------------------> skip FIFO -----+-> Concat -> Thr_act

* **Frame budget F = 81920 cycles = the MAXPOOL FLOOR**, 1.25 * 256 * 256 (FINN's `get_exp_cycles` estimate for the 2D `StreamingMaxPool`: one cycle per input pixel plus one per
  output pixel, no PE/SIMD). So the target is **5.0 cyc per OUTPUT pixel** (16384 output pixels), 1.25 per input pixel. Measure as in section 10.1 with
  `pixels_per_frame = 16384`. **Please report the maxpool's real rtlsim behaviour**: if its output write overlaps the input read the real floor is 65536 cycles
  (1.0 / pixel) and the whole network budget could drop by 20%; run this probe first among the three and look at the stable period (`(cycles - latency)/(N-1)`).
* One input channel means no channel parallelism: Thr_in, Dup, FMPad and the maxpool are all pixel-serial (>= 65536 cycles). MVAU_c has MW = 9, MH = 3: the model picks
  PE 1 x SIMD 9 (parallel_window = 1 on the SWG, SIMD of the FMPad/SWG = 1 = Cin), 9 DSP at every bit width; block = 1.66k LUT / 5 BRAM18 / 9 DSP at INT4
  (INT6 1788 LUT, INT8 2244 LUT). Simulated steady 5.02 cyc/px (maxpool at 100% of F, MVAU_c at 60%), first-out latency ~540 cycles; join FIFOs: skip FIFO (maxpool end)
  96 words x A bit, FIFO main (conv end) 77 words; FIFO in front of the 3x3 FMPad ~260 words (next-frame prefetch, W+2 pixels).
* The stride-2 3x3 window never reads the last padded row / column (258 padded, last window ends at 256): the sim's window node drains them before the next frame -- a real SWG
  does this too; if its throughput looks lower than the model, check the SWG's frame-boundary handling first.
* Roles (`identify_roles_init`): thr_in, dup, fmpad, swg, mvau_c, thr_c, maxpool, thr_m, concat, thr_act. thr_c / thr_m are the shared `branch_quant` thresholds before the
  concat, thr_act the BN + ReLU threshold after it -- streamlining may move the BN scale differently; report what the landed graph looks like if the roles do not resolve.
  The concat path needs the repo's existing concat fixups (`MoveTransposePastJoinConcat`, `MoveScalarMulPastConcat`, see finn_enet_build*.py); no new FINN pass is required.


## 13. Final-deconvolution probes (6 cases): `fnl_cin4_cout5_in128_int{4,6,8}_{bias,nobias}`  (`SET=fnl`)

ENet U4 final layer of `LayerQuantEnetFINN`: `qnn.QuantConvTranspose2d(c5 = 4 -> out_channels = 5, kernel 2, stride 2)`, 128x128 input -> 256x256 output, Int8 weight quantizer at
weight_bit_width = b. Model: `MILP/analytical/fnl_block.py` (+ `fnl_block_sim.py`, `test_fnl_block.py`, section in `analytical.md`). **Not a bottleneck**: one path, no Dup, no skip, no join
FIFO, no output threshold (the output is the raw logit). Export: `export_fnl_probe.py`; folding: `make_fnl_folding_configs.py`.

    Thr_in -> FMPadding_Pixel -> SWG (2x2, stride 1) -> MVAU_f (4*Cin = 16 -> Cout = 5) -> [Bias add]

* Frame budget F = 73728 (same as the other U4 probes) = **1.125 cyc per OUTPUT pixel** (65536 output pixels; 4.5 per input pixel). Measure as in 10.1 with `pixels_per_frame = 65536`
  (`compare_probe_vs_model.py` does this for `block == "final"`). This is the tightest budget of all probes: one output pixel per cycle is the floor (65536), and FMPadding_Pixel emits the
  (2H+1)(2W+1) = 257 x 257 = 66049-cycle zero-inserted image with SIMD = Cin. Expect the stable period at ~1.01 cyc/px (model), i.e. 88% of the budget.
* Folding (all bit widths): FMPadding_Pixel SIMD 4, SWG SIMD 4 (parallel_window 0), MVAU_f PE 5 x SIMD 16 (80 MACs/cycle -> 48 DSP, one output pixel per cycle), bias PE 5. The lowered
  transposed conv spends 4x MACs on inserted zeros (3 of the 4 window elements), as in the up-blocks.
* Two variants, because the repo itself is inconsistent here -- **this is the main open question of this block**:
  * `bias`   = `LayerQuantEnetFINN` default (`final_bias=True`, `Int32Bias`): the graph has an integer bias add after the MVAU. Expected to land as a `ChannelwiseOp` (role `bias`).
  * `nobias` = what `hardware/finn_enet_prod_export.py` exports today (comment there: "bias in ConvTranspose requires extra BN/threshold handling").
  Report for `bias`: what node the bias becomes (ChannelwiseOp, an MVAU bias attribute, or a leftover `Add`/`Mul` that stays OUTSIDE the dataflow partition). If it does not lower, say so --
  that decides whether the production network must keep `final_bias=False` or needs a new absorb pass. `identify_roles_final` raises "bias variant: no node after the MVAU" in that case.
* The output is tagged INT8 (as `finn_enet_prod_export.export_model` does). Streamlining may leave the weight-scale `Mul` after the MVAU outside the dataflow partition: that is fine for
  these probes (the partition should end at the MVAU / bias node); report it.
* Roles (`identify_roles_final`): thr_in (producer of the FMPadding_Pixel; its PE is set equal to the FMPadding SIMD), fmpadpix, swg_u, mvau_f, [bias].
* **Expected resources** (INT4 / INT6 / INT8): bias LUT 2276 / 2395 / 2276, nobias LUT 1548 / 1647 / 1508, BRAM18 0, DSP 48 for all (FMPadding_Pixel and ChannelwiseOp LUT are not calibrated;
  the bias add is priced like an AddStreams -- treat the ~730 LUT gap bias vs nobias as a placeholder, report the real number). Simulated first-out latency ~260 cycles, FIFO depths 2-4 everywhere
  except a 4-word prefetch FIFO before FMPadding_Pixel. Nothing here should deadlock; if it does, it is the FMPadding_Pixel / SWG handshake at SIMD 4 (report which FIFO is full).

## 14. Pool-route probes (2 cases): `init_cin1_cout4_in256_int4_pool`, `dn_cin16_cout32_in64_int4_mvau_pool`  (`SET=pool`)

FINN's 2D `StreamingMaxPool` is not foldable (1.25 * H * W cycles). FINN also has a second route: `InferPool` lowers a MaxPool to a depthwise SWG (`Im2Col`) + `Pool_hls` (`PE`), cycles
(C * K^2 / PE) * OH * OW, which can be folded over channels. The probe script monkeypatches `convert_to_hw_layers.InferStreamingMaxPool = InferPool` when `probe["pool_impl"] == "pool"`.
Each pool probe **shares the ONNX** of its StreamingMaxPool twin; only the folding json (extra rows `swg_p`, `pool`) and the `_probe.json` (`pool_impl: "pool"`) differ.
* **First check (gate 0 for this set):** the container FINN is `v0.10.1-10g39f0c9a6b`; confirm `InferPool` (or `InferPool_Batch`) exists in
  `finn/transformation/fpgadataflow/convert_to_hw_layers.py` and which op names result (`Pool_hls`? `Pool_rtl`? `ConvolutionInputGenerator` with `depthwise = 1`). If neither exists, stop and report;
  do not invent a pass. Run `STOP_AFTER=convert` first and list the nodes.
* `init_cin1_cout4_in256_int4_pool`: **F = 69632** (T_out = 4.25 per output pixel), not 81920: at F = 65536 the depthwise SWG (66050 cycles) and the conv-branch FMPadding (66564) are infeasible, so
  69632 is the first sensible budget. Folding: `swg_p` SIMD 1 + `pool` PE 1 (Cin = 1, so no channel folding here; the gain over StreamingMaxPool is that Pool_hls reads 1 pixel per cycle,
  no 1.25 factor). Model: steady 4.06 cyc/px, latency 536, 1750 LUT / 3 BRAM18 / 9 DSP, skip FIFO 68 words.
* `dn_cin16_cout32_in64_int4_mvau_pool`: F = 73728 (T_out = 72), `swg_p` SIMD 2 + `pool` PE 2 (PE 1 = 80.9k cycles does not fit). Model: steady 72.0, latency 3774, 4193 LUT / 1 BRAM18 / 28 DSP, skip FIFO 1421 words.
* **Report:** real rtlsim period of the Pool route vs the StreamingMaxPool twin (init: 4.06 predicted vs 5.02 for the streaming route; dn: unchanged 72), the landed `Pool` PE and `swg_p` SIMD (does
  `parallel_window` / `depthwise` constrain SIMD == PE?), and LUT/BRAM of the Pool node. Also **report the StreamingMaxPool twin's real cycles per frame** (init, first run of `SET=int`): the model
  assumes 1.25 * H * W = 81920; if the rtlsim shows 65536 the whole network budget could drop.

## 15. Batch protocol and per-bottleneck change log (READ BEFORE LAUNCHING ANYTHING)

### 15.1 Run EVERY probe through build + OOC synthesis, at most 4 builds at a time
All 41 probes (reg 15, dn 6, up 6, up5 3, int 3, pool 2, fnl 6) need the **full flow: stitched-IP rtlsim AND out-of-context synthesis**, not rtlsim only (this supersedes the "rtlsim first, OOC secondary"
wording of section 1 and the single "--ooc later" step of section 4.6). One run per case does both: `EXTRA="--ooc"`.
* **Never more than 4 builds running at once** across the whole container (Vivado/HLS memory). `run_probes.sh` caps one SET at `JOBS=4` (default) -- **do not raise `JOBS`, and do not start a second
  `run_probes.sh` while one is running** (two sets in parallel = 8 builds). Run the sets one after another.
* Suggested order (small/new first, so a script bug does not burn a 15-case batch): `SET=fnl` (6) -> `SET=int` (3) -> `SET=pool` (2) -> `SET=up5` (3) -> `SET=up` (6) -> `SET=dn` (6) -> `SET=reg` (15, d16 last).
  First `STOP_AFTER=folding` for one case of each SET (gate 3), then the full batch: `CONTAINER=<name> SET=fnl EXTRA="--ooc" JOBS=4 bash run_probes.sh`.
* After **each** set: `CONTAINER=<name> bash collect_probe_outputs.sh` (copies `probe_result.json`, `report/` and `utilization_placed.rpt` out of the container; the OOC json DSP field is wrong, the raw
  report is the truth) and `python3 compare_probe_vs_model.py`. Do this before starting the next set -- build dirs under `/tmp/finn_dev_*` are lost when the container is recreated.
* Per-case timeouts: `TIMEOUT` default 14400 s; exit code 124 = timeout = possible deadlock (capture which FIFO is full). reg d16 and fnl (66k-cycle frames x 3) are the longest rtlsims -- raise `TIMEOUT`
  rather than `JOBS`.

### 15.2 What changed per bottleneck since you last saw it (RERUN = results you already have are stale, NEW = never built)
* **reg (15, `bottleneck_cin32_d{1,2,4,8,16}_int{4,6,8}`): RERUN.** FIFO sizing now escalates through a capped schedule (d16 no longer asks for 17k-32k word FIFOs), the 3-frame saturated run is the
  sizing criterion, a `tol_soft` 3% warning exists, and the prediction carries FIFO memory choices (`mem`, `depth_alloc`, BRAM18/URAM/LUT per FIFO: srl <= 64 deep, bram pow2, uram if efficient). The forced
  depths in `_folding.json` were regenerated -> **earlier regular-probe results used old depths; rerun all 15**. `stable_throughput` from FINN is miscaled (divides by N, not N-1);
  `compare_probe_vs_model.py` recomputes it, trust that column.
* **dn (6, `dn_cin16_cout32_in64_int{4,6,8}_{fmpad,mvau}`): RERUN + pool twin.** Default (StreamingMaxPool) structure unchanged, but the model gained `pool_impl` and the multi-frame prefetch FIFOs
  (before FMPad / SWG) were re-sized -> regenerated folding json. New: the Pool-route twin (section 14). The `fmpad` variant still needs your `finn_channel_pad.py` pass (stub in repo).
* **up (6, `up_cin32_cout16_in32_*`) + up5 (3, `up_cin16_cout4_in64_*_conv`): NEW** (sections 11 / 11.1). `identify_roles_up` classifies 1x1 MVAUs by what follows; no new FINN pass.
* **int (3, `init_cin1_cout4_in256_int{4,6,8}`) + int pool (1): NEW** (sections 12 / 14). Needs the concat fixups already in the repo; the strided 3x3 SWG's frame-boundary behaviour is the thing to watch.
* **fnl (6): NEW** (section 13). New role dispatch `identify_roles_final`; the `bias` variant is an open FINN-lowering question.
* **Shared script changes** (`finn_bottleneck_probe_build.py`): `identify_roles` dispatches on `probe["block"]` in {down, up, init, final}; `thr_in` PE is taken from the Dup, or from the
  FMPadding_Pixel SIMD when there is no Dup (final); MVAU divisibility asserts run over whichever `mvau_*` roles exist; Pool-route monkeypatch. `run_probes.sh` knows
  `SET=reg|dn|up|up5|int|pool|fnl`; `collect_probe_outputs.sh` and `compare_probe_vs_model.py` know all prefixes and `block == "final"`. `run_probes.sh` re-`docker cp`s everything on every launch --
  do not hand-edit the container copies.

## 16. Diagnostic probes for the slow `noconv` upsampler (6 cases, INT4 only): `up_cin32_cout16_in32_int4_noconv_{pe2,pe4,fjoin,fupnn,fall,pe4fall}`  (`SET=upd`)

From FINN_AGENT_FINAL_REPORT.md: `up_cin32_cout16_in32_int{4,6,8}_noconv` landed every PE/SIMD as predicted but measure 20.3-20.5 cyc/px (T = 18, model 17.6); the `conv` twin passes at 18.0.
`noconv` is the decoder the network actually uses, so this is the next thing to explain. The six twins share the ONNX of `..._int4_noconv`; only the folding json differs
(`make_up_diag_configs.py` writes them; each json has a `diag` block listing exactly what was overridden). Run them as a normal set (build + OOC, JOBS <= 4, section 15.1), **after** re-running
the baseline `up_cin32_cout16_in32_int4_noconv` once so everything uses the same container state.

| Probe | Change vs baseline | Tests |
|---|---|---|
| `_pe2`, `_pe4` | `Thr_s`, `Add`, `Thr_out` at PE 2 / 4 (baseline PE 1: 16 cyc/px = 89% of F, the `UpNN -> Thr_s` DWC is 64 -> 4 bit) | the join chain / its DWC has no slack at PE 1 (per-word overhead, bubbles) |
| `_fjoin` | ext-side join FIFO `Thr_e -> Add` 17 -> 512 words (FINN's stock sizing wanted 484; ours was 17) | `Add` waits on the main branch while ext data piles up |
| `_fupnn` | FIFO in front of `UpsampleNearestNeighbour` (`dwc -> upnn`) 8 -> 40 words (one input row), FIFO after it 8 -> 91 (FINN stock) | `mvau_p` cannot run ahead while the upsampler re-emits a row |
| `_fall` | `_fjoin` + `_fupnn` | the two FIFO effects together |
| `_pe4fall` | PE 4 + both FIFO changes | ceiling: if this is still > 18.5 the limiter is not in these suspects |

**Reading the results (measure with (cycles - latency_cycles)/(N-1)/4096; target 18.0):**
* a probe that reaches <= 18.4 names a cause; run the single-change probes before drawing conclusions from `_fall` / `_pe4fall`;
* if only the PE probes recover, the join chain (DWC / Thresholding / AddStreams at PE 1) needs slack: report the real cycles per word of the 64 -> 4 bit DWC;
* if only `_fjoin` / `_fupnn` recover, the join FIFO sizing in `up_bottleneck_sim.py` is wrong for `noconv` and the model needs fixing (report the max occupancy of `thr_e -> add` and `thr_s -> add`);
* if nothing recovers, report per-node cycle counts of the stitched model (`AnnotateCycles` / per-node `get_exp_cycles` versus the rtlsim) and which FIFO is full or empty at the end of a frame.
Also report `stock` vs `forced` depth per edge (the usual `stages.fifo`) and the real max occupancy of the join FIFOs. The model's own numbers for these twins are in each json (`predicted`).

## 17. Result of section 16 and what to run next (PE of the nodes behind the nearest-neighbour upsampler)

Measured (INT4 `up_cin32_cout16_in32_..._noconv`, rtlsim cyc per OUTPUT pixel, target 18.0): baseline 20.53, `_fjoin` 20.53, `_pe2` 17.96, `_pe4` 16.25, `_fupnn` 16.00, `_fall` 16.00, `_pe4fall` 16.25.
The join FIFO was not the cause; the nearest-neighbour upsampler (read a row, emit each pixel twice, then re-emit the row with no input) needs either a faster drain behind it (PE of `Thr_s` / `Add` /
`Thr_out`) or buffering around it. Decision: **raise the PE** (the model now derives it: `up_bottleneck.model_up_bottleneck(join_pe=None)`, rule and worked numbers in `analytical.md`, section
"PE of the nodes behind the nearest-neighbour upsampler").

**The PE of that join chain must scale with the target.** `PE >= 2 * Cout / (t_in - c_p)` (t_in = 4 * T_out cycles per input pixel, c_p = MVAU_p cycles per input pixel, rounded up to a divisor of Cout).
If a later experiment tightens the target (smaller F / T_out) the same node needs a larger PE, or buffering when `t_in - c_p < 2` (the model prints a warning). Do not copy PE 4 to other sizes: regenerate the folding
(`make_up_folding_configs.py`) so the PE follows the rule.

Changed files and what to rebuild:
* `inputs/up_cin32_cout16_in32_int{4,6,8}_noconv_folding.json` were regenerated: Thr_s / Add / Thr_out at PE 4 (before: PE 1). The INT4 json is identical to the validated `_pe4` twin, so **no INT4 rebuild is
  needed**; the old `up_..._int4_noconv_merged_*` result folders are the superseded PE 1 configuration (20.53). **Rebuild INT6 and INT8 `noconv` (2 builds, rtlsim + OOC)** with the new jsons; expect ~16.3 cyc/px.
* The 6 diagnostic twins stay as they are (`make_up_diag_configs.py` passes `join_pe` explicitly, so `_fjoin`, `_fupnn`, `_fall` still mean PE 1 + their FIFO change).
* Not changed: the `conv` decoder (join chain stays PE 1; it passes), the FIFO sizing rule in `verify_with_sim` (still shrinks to the smallest depth meeting T; see analytical.md for why that is risky).
* Up5 (16 -> 4, 64x64 in) `noconv`, once exported, gets its PE from the same rule (t_in 18, c_p 16 -> PE = Cout = 4).

## 18. Initial block with the landed graph order: `init_cin1_cout4_in256_int{4,6,8}_thrpre`  (`SET=intpre`, 3 builds, rtlsim + OOC)

Why: the first init probes (`init_cin1_cout4_in256_int{4,6,8}`) measured 15.78 cyc/px against the 5.0 target. Their FIFO report showed the hardware graph is `Dup -> Thr_m -> MaxPool -> Concat`:
FINN's streamline step `MoveMaxPoolPastMultiThreshold` (`hardware/finn_enet_build.py`) swapped the exported `MaxPool -> Quant` into `Thr -> MaxPool`. The model had `Dup -> MaxPool -> Thr_m -> Concat`,
so (a) Thr_m's work was priced at 16,384 pixels instead of 65,536 and (b) **the 96-word skip FIFO was forced on the edge `thr_m -> concat`, which does not exist in hardware**; in the first
probes the real join edge `maxpool -> concat` kept FINN's 2 words and the edges in front of the maxpool kept FINN's stock sizes (25,304 and 65,536 words). That mismatch is a candidate cause of the slowdown.

What changed (model `int_bottleneck.py` + `int_bottleneck_sim.py`, 16 tests pass): Thr_m is upstream of the pool (65,536 pixels, PE 1, 4.0 cyc per output pixel = 80% of F), new DWC edges
`Dup->Thr_m`, `Thr_m->pool`, the skip FIFO sits at the pool output (`maxpool -> concat`, 96 words x 4 bit, forced; `dup -> thr_m` and `thr_m -> maxpool` 2 words). The simulated rate is unchanged (5.02 cyc/px,
maxpool at 100% of F); folding PE/SIMD and totals are unchanged (1658 / 1788 / 2244 LUT, 5 BRAM18, 9 DSP for INT4 / 6 / 8).

The new cases `init_cin1_cout4_in256_int{4,6,8}_thrpre` share the ONNX of the old ones; only `_folding.json` and `_probe.json` (`thr_m_order: "pre"`) differ. The old folding jsons and result folders are the
earlier (post-pool) model and are left as they are. `identify_roles_init` already resolves `thr_m` as the producer of the maxpool (no script change needed).

**Please report, per case:** rtlsim cyc per OUTPUT pixel ((cycles - latency_cycles)/(N-1)/16384, target 5.0); forced vs stock depth for `dup -> thr_m`, `thr_m -> maxpool`, `maxpool -> concat`; and the max occupancy of
`maxpool -> concat`. If the rate stays near 15.8 the FIFO mismatch was not the cause: then report the per-node busy / stall counts of `StreamingMaxPool_hls` and the Dup, and whether the real maxpool takes
about 4 cycles per input pixel (the first probes' frame period was ~258.6k cycles = 3.95 per input pixel). The Pool-route twin (`init_..._int4_pool`) still has the old order; regenerate it with
`make_int_folding_configs.py --pool-impl swg_pool --F 69632` after this set, if the result calls for it.
