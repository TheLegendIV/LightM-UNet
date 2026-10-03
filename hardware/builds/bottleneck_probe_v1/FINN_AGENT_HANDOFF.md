# Handoff: bottleneck probe builds (for the agent that runs on the FINN container)

You are taking over a prepared set of build files. Everything host-side is done and tested; the FINN-side script has
**never run** (no FINN container was available when it was written). Your job is to make it run, run it, and report.
Read `AGENTS.md` (repo root) first: FINN code must be run as `docker exec -e HOME=/tmp/home_dir <container> ...`, files are
`docker cp`'d flat into `/home/thelegendiv/finn/notebooks/enet/`, Vivado needs `source /tools/Xilinx/Vivado/2022.2/settings64.sh`.

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
