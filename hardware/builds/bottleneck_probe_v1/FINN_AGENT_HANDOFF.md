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
