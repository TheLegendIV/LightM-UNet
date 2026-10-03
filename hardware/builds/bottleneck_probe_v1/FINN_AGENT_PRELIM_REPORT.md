# Bottleneck Probe v1 — Preliminary Findings (2026-10-03)

Status snapshot while work is still in progress — see "Open / in-flight" at the
bottom for what's not yet done. This probes whether the tight FIFO/folding
config in `FINN_AGENT_HANDOFF.md` deadlocks in real FINN rtlsim, case by case
across dilation {1,2,4,8,16} x int{4,6,8}.

## Environment

- Container: `finn_persistent` (run everything as `docker exec -e HOME=/tmp/home_dir finn_persistent ...`).
- Build script: `finn_bottleneck_probe_build.py`, deployed flat into
  `/home/thelegendiv/finn/notebooks/enet/` alongside all 15 cases'
  `<case>.onnx` / `<case>_probe.json` / `<case>_folding.json`.
- Vivado 2022.2 sourced via `/tools/Xilinx/Vivado/2022.2/settings64.sh` before
  every build.

## Bugs found and fixed this round

All five fixes below are already landed in `finn_bottleneck_probe_build.py`
and `hardware/checks/test_compose_thresholds.py` (host + container copies in
sync).

1. **Opset mismatch in the threshold-merge unit test (gate 2).** onnx 1.17.0
   defaults new models to opset 22; onnxruntime 1.18.1 only supports <=21,
   causing `InvalidArgument` in `execute_onnx`. Fixed by pinning
   `opset_imports=[onnx.helper.make_opsetid("", 11)]` in `build_graph()`.
2. **`QuantIdentityHandler` rejects unsigned identity `Quant` nodes (gate 3).**
   FINN's qonnx->finn converter hard-requires `signed=1` on any Quant node
   with no producer (our `thr_in` stand-in for the previous block's ReLU
   output). Fixed with `step_force_signed_identity_quant` (force `signed=1`
   before conversion) + `step_fix_signed_thresholds` (downgrade the resulting
   `MultiThreshold` back to unsigned afterwards whenever `out_bias >= 0`,
   i.e. the comparator never needed negative range).
3. **DWC insertions silently broke FIFO-depth forcing (gate 4).** The
   predicted-fifos list brackets every `StreamingDataWidthConverter` with two
   separate predicted entries (`role->dwc`, `dwc->role`). The original
   `SKIP_OPS` treated DWC as transparent when walking producer/consumer
   chains, collapsing both flanking FIFOs onto the same key and leaving half
   the FIFOs (including the single most important one, the FMPad feed) at
   FINN's stock depth with no error raised. Fixed by removing
   `StreamingDataWidthConverter` from `SKIP_OPS`.
4. **Vivado FIFO-Generator IP rejects our small forced depths (gate 5).**
   FINN auto-promotes any FIFO whose natural stock depth is large to
   `impl_style="vivado"` (the Xilinx FIFO Generator IP, which only accepts a
   fixed power-of-2 depth menu: 16, 32, 64, ... 32768). Forcing a small depth
   (e.g. 4) onto one of these crashed `CreateStitchedIP` with `ERROR:
   [IP_Flow 19-3461] Value '4' is out of range`. Fixed by forcing
   `impl_style="rtl"` (plain SRL-based FIFO, any depth valid) whenever we
   override a FIFO's depth — also required anyway since rtlsim only supports
   `impl_style="rtl"` per an explicit assert in `streamingfifo_rtl.py`.
5. **Stale generated module names collided after node renumbering (gate 5).**
   `SplitLargeFIFOs()` can insert new FIFO nodes, and the subsequent
   `GiveUniqueNodeNames()` then renumbers every `StreamingFIFO_rtl_N` node in
   graph order. Any FIFO whose code was reused ("pre-existing") rather than
   regenerated still had its *old* name baked into its generated Verilog
   `module` declaration, so two different cached directories could both
   declare `module StreamingFIFO_rtl_7`, and Vivado's block-design cell
   creation failed with `[filemgmt 56-190] Failed to uniquely resolve
   reference. Multiple different modules were found that match the name...`.
   Fixed by unconditionally calling `reset_implementation()` on every
   `StreamingFIFO*` node right after the post-split rename, before
   `PrepareIP`/`HLSSynthIP` regenerate IP.

## Gate 2-4 results (reference case `bottleneck_cin32_d8_int4`)

- Gate 2 (merge-pass unit test): 11/11 passing.
- Gate 3 (folding): landed PE/SIMD per role matches `FINN_AGENT_HANDOFF.md`
  Section 8 reference facts exactly.
- Gate 4 (FIFO depths): all 18 FIFO edges matched predicted depths exactly
  (feed-`MVAU_r`=4, feed-`FMPad`=14, skip=8736, every other edge=2); only the
  2 graph-boundary FIFOs (pre-`thr_in`, post-`thr_out`) are intentionally
  unmatched (no prediction exists for I/O-boundary FIFOs).

## Gate 5 (full build -> stitched IP -> multi-frame rtlsim)

First full run used only 3 frames and reported steady_cyc_per_pixel=58.6,
beating the 72 cyc/px target by more than expected — flagged as possibly a
transient-regime artifact given the large (8736-deep) skip FIFO needs more
than 2 "extra" frames to reach true steady state. Reran with
`--rtlsim-frames 6` for all cases going forward.

### Wave 1 (`d1_int4`, `d1_int6`, `d1_int8`, `d2_int4`) — 6 frames, rtlsim only

**4/4 PASS.** Steady-state cyc/px: 61.3-63.2, comfortably under the 72
target for all four. No deadlocks; `Number of inputs consumed` ==
`Number of outputs produced` (196608/196608) for every case.

| case | steady cyc/px | target |
|---|---|---|
| d1_int4 | 61.39 | 72 |
| d1_int6 | 61.37 | 72 |
| d1_int8 | 61.34 | 72 |
| d2_int4 | 63.17 | 72 |

**Caveat on "latency rtl/pred" column** in `compare_probe_vs_model.py`'s
output: this compares two different quantities, not a resource or
correctness check. FINN's `latency_cycles` field is measured from a
*separate single-frame (batch=1) run*'s total cycle count, which is
dominated by `~1023 pixels * cyc_per_pixel` (roughly constant across
dilation, ~76-78k cycles observed for all of wave 1) — not the "first pixel
out" pipeline-fill latency the analytical model predicts (which scales with
padding/dilation: 2686/5131/10012/19780 cycles for d1/d2/d4/d8). The verdict
function only gates on steady_cyc_per_pixel, so this mismatch does not
affect pass/fail.

### Wave 2 (`d2_int6`, `d2_int8`, `d4_int4`, `d4_int6`) — 6 frames, rtlsim only

Still running at time of writing.

### Resource usage (LUT/BRAM18/DSP)

Not yet available for any case — requires `--ooc` (out-of-context
synthesis), which the script only runs as part of the same pass as rtlsim
(no resume-from-stitched-IP option, so getting resource numbers means a full
rebuild with `--ooc` added). **OOC reruns for wave 1's 4 cases are currently
in flight** (launched alongside wave 2, 8 total parallel builds on the
container by explicit request, exceeding the previously-used 4-parallel
cap). Real synthesis is expected to take noticeably longer than rtlsim
alone; results pending.

## Known stale artifacts

`finn_deployment_outputs/bottleneck_cin32_d8_int4_merged_20261003_140404`
through `..._142933` are leftover from the original fix-finding iterations on
the reference case (pre-fix crashes, one incomplete `probe_result.json`-less
dir). The valid, fully-passing single-case run is
`..._143857`. These stale dirs are harmless but will appear in a raw
`ls`/`collect_probe_outputs.sh` sweep; safe to ignore or clean up later.

## Open / in-flight

- Wave 2 rtlsim (4 cases): running.
- Wave 1 OOC resynthesis (4 cases): running, for LUT/BRAM18/DSP numbers.
- Not yet started: `d4_int8`, `d8_int6`, `d8_int8`, `d16_int4`, `d16_int6`,
  `d16_int8` (6 cases remaining of the full 15; `d8_int4` already validated
  separately as the reference case).
- Not yet started: negative control (`--skip-scale 0.5`, expected to
  deadlock/timeout — proves the harness can detect real failures) and
  stock-FIFO control (`--fifo-policy stock`).
- Not yet started: `collect_results.py` / calibration-database integration
  for any of these numbers.
