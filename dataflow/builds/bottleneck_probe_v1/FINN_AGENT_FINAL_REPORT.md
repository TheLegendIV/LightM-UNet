# Bottleneck Probe v1 — Final Report (2026-10-04)

Supersedes `FINN_AGENT_PRELIM_REPORT.md` (that doc is "in progress"; this one
is the end state after the full batch finished and every probe's output was
pulled back from the container). All raw data is under `results/<case>_merged_<ts>/`
(`probe_result.json` + `report/` + `utilization_placed.rpt` where OOC ran).
Re-run `python3 compare_probe_vs_model.py` any time to regenerate the verdict
table below from the raw JSONs (`PASS` = rtlsim steady <= T*1.02, `NEAR` <=
T*1.03, `SLOW` otherwise, `NO-RTLSIM` = build never reached rtlsim).

Where a case has multiple timestamped folders (re-run after a fix during the
session), only the **latest timestamp** is the authoritative result — earlier
ones are kept on disk for history but are superseded.

## Final verdict per case (latest run only)

| Case | Verdict | rtlsim cyc/px | T (target) | OOC ran? |
|---|---|---|---|---|
| bottleneck_cin32_d1_int{4,6,8} | PASS | 72.00 | 72.0 | yes |
| bottleneck_cin32_d2_int{4,6,8} | PASS | 72.14 | 72.0 | yes |
| bottleneck_cin32_d4_int{4,6,8} | PASS | 72.18 | 72.0 | yes |
| bottleneck_cin32_d8_int{4,6,8} | PASS | 72.62 | 72.0 | yes |
| bottleneck_cin32_d16_int{4,6,8} | NEAR | 73.88 | 72.0 | yes |
| dn_cin16_cout32_in64_int{4,6,8}_mvau | PASS | 72.00 | 72.0 | yes |
| dn_cin16_cout32_in64_int4_mvau_pool | PASS | 72.00 | 72.0 | yes |
| dn_cin16_cout32_in64_int{4,6,8}_fmpad | **dead end** (see below) | — | — | never ran |
| up_cin16_cout4_in64_int4_conv | PASS (rtlsim) | 4.49 | 4.5 | **no — RTL bug** |
| up_cin16_cout4_in64_int6_conv | PASS | 4.43 | 4.5 | yes |
| up_cin16_cout4_in64_int8_conv | PASS (rtlsim) | 4.36 | 4.5 | **no — RTL bug** |
| up_cin32_cout16_in32_int4_conv | PASS (rtlsim) | 18.02 | 18.0 | **no — RTL bug** |
| up_cin32_cout16_in32_int6_conv | PASS | 18.06 | 18.0 | yes |
| up_cin32_cout16_in32_int8_conv | PASS (rtlsim) | 18.06 | 18.0 | **no — RTL bug** |
| up_cin32_cout16_in32_int4_noconv | **SLOW** | 20.53 | 18.0 | yes |
| up_cin32_cout16_in32_int6_noconv | **SLOW** | 20.43 | 18.0 | yes |
| up_cin32_cout16_in32_int8_noconv | **SLOW** | 20.34 | 18.0 | yes |
| init_cin1_cout4_in256_int{4,6,8} | **SLOW** | 15.78 | 5.0 | yes |
| init_cin1_cout4_in256_int4_pool | **SLOW** | 5.01 | 4.2 | **no — Vivado crash** |
| fnl_cin4_cout5_in128_int4_bias | **NO-RTLSIM** (never retried) | — | 1.1 | — |
| fnl_cin4_cout5_in128_int{4,6,8}_nobias | **SLOW** | 763.95 | 1.1 | yes |

## Known root causes (already in `/memories/repo/finn_gotchas.md`, unchanged)

1. **`dn_*_fmpad`** — confirmed dead end: the skip branch stays NCHW end to
   end, FINN never brackets a standalone `Pad` with Transpose the way it does
   `Im2Col`, and a hand-inserted Transpose sandwich breaks dataflow
   partitioning. Use the `mvau` variant (already PASS at int4/6/8) instead —
   do not re-attempt.
2. **RTL MVU Verilog bug** — OOC synth of the `up`/`up5` `conv` variant at
   int4/int8 (not int6) fails in Vivado with a part-select-out-of-range error
   inside FINN's own `mvu_4sx4u.sv`/`mvu_8sx8u_dsp48.sv`. rtlsim itself PASSes
   fine (table above) — only the Vivado OOC step is blocked. No fix yet;
   needs either a different PE/SIMD folding or a FINN-side RTL patch.
3. **`init_cin1_cout4_in256_int4_pool`** (and, confirmed 2026-10-04, ALL 3
   `SET=intpool` InferPool-route twins `pool_thrpre`/`poolpw_thrpre`/
   `poolpw81_thrpre` — 3/3 new cases, same crash, same stage) — OOC synth hits a deterministic
   Vivado-internal Tcl-stack crash (`TclStackFree: incorrect freePtr`) during
   Technology Mapping. Not fixed; not retried since.
   **ROOT CAUSE NARROWED 2026-10-04** via the real `hs_err_pid*.log` (not the
   top-level `runme.log`): every individual module, including `Pool_hls_0`
   itself, synthesizes cleanly with no errors — FSM inference, XDC
   constraints, timing optimization, and the ROM/RAM/DSP mapping reports all
   print successfully. The crash fires on the very first line of the next
   global stage ("Start Technology Mapping"), before any real mapping work
   happens. The native crash stack is entirely inside Vivado's own
   closed-source Tcl task-management code, not our RTL/HLS:
   `Tcl_Panic -> TclStackFree -> ... -> HRTInvoker::inProcessLaunch ->
   librdi_vivadotasks.so/librdi_designutils.so` — Vivado launching some
   internal report/task in-process (likely the resource-report generator for
   Technology Mapping, same class as the ROM/RAM tables it just printed)
   whose Tcl interpreter state is already corrupted (`incorrect freePtr` =
   double-free-class bug) before it can execute. 100% correlated with
   `Pool_hls` being present in the flattened OOC design — every `_thrpre`
   case *without* `Pool_hls` reaches synth fine (just SLOW), every case
   *with* it (4/4 now) crashes identically.
   **CORRECTED 2026-10-04 (was wrongly called unfixable/opaque)**: this
   matches a documented, known Vivado crash class, not a one-off. AMD's own
   "Vivado Synthesis Crash Debugging Guide" (KB 946862) describes the
   identical signature -- `Abnormal program termination (6)` +
   `hs_err_pid*.log` -- under "Crash in the Cross Boundary Optimization
   Phase" (our crash is one stage later, "Start Technology Mapping", but
   same family: Vivado's default `-flatten_hierarchy rebuilt` tries to
   flatten/optimize across module boundaries around this point, and BRAM
   inference specifically is called out as a common trigger -- plausible
   here since `Pool_hls` likely uses an internal line buffer/BRAM).
   Untried, AMD-documented workarounds, in order of invasiveness:
   1. `-max_bram 0` on the synth run (tests whether BRAM inference for
      `Pool_hls`'s line buffer is the trigger).
   2. `KEEP_HIERARCHY`/`DONT_TOUCH` on just the `Pool_hls` instance (blocks
      cross-boundary opt into/out of it without a global QoR hit).
   3. `-flatten_hierarchy none` globally (blunt fix, costs QoR).
   4. Toggle `Pool_hls`'s IP output-products synthesis mode (OOC vs
      Global) -- AMD notes designs failing in one mode pass in the other.
   5. `vivado -stack 2000` (rules out plain Tcl stack overflow; `hs_err`
      didn't show a stack-depth message so lower priority).
   None of these have been attempted yet -- this is the next concrete step
   before concluding the InferPool route is a dead end.
   **TESTED 2026-10-04: workaround 1 (`-max_bram 0`) FAILED.** Uncommented
   `set_property "steps.synth_design.args.max_bram" "0" $obj` in the
   container's `oh-my-xilinx/vivadocompile.tcl` and reran
   `init_cin1_cout4_in256_int4_pool_thrpre` (rtlsim passed, 246078 cycles).
   OOC synth hit the IDENTICAL crash at the IDENTICAL stage (`Start
   Technology Mapping` -> `TclStackFree: incorrect freePtr` -> `Abnormal
   program termination (6)`). Confirmed BRAM inference is NOT the trigger:
   with `max_bram=0` active, the RAM mapping report shows only Distributed
   RAM entries (zero BRAM table rows at all), yet the crash is unchanged.
   Rules out hypothesis 1 cleanly. Next: try workaround 2
   (`KEEP_HIERARCHY`/`DONT_TOUCH` on just the `Pool_hls` instance) or 3
   (`-flatten_hierarchy none` globally).
   **TESTED 2026-10-04: workaround 3 (`-flatten_hierarchy none`) also
   FAILED.** Same exact crash signature, but at a DIFFERENT point: with
   hierarchy flattening disabled, Vivado retimes each module individually
   in a loop, and the crash now fires mid-loop, right after
   `finn_design_FMPadding_rtl_0_0` finishes retiming -- BEFORE `Pool_hls_0`
   itself is ever reached in that per-module retiming sequence. This
   complicates (without refuting) the `Pool_hls`-presence correlation: the
   crash isn't tied to the moment `Pool_hls_0` itself gets processed, it
   happens at a different point depending on which synthesis strategy is
   used, suggesting something more like a cumulative/order-dependent Tcl
   interpreter corruption than a `Pool_hls`-specific operation triggering
   it directly. Next to try: workaround 2 (`KEEP_HIERARCHY`/`DONT_TOUCH` on
   just the `Pool_hls` instance -- needs a Verilog attribute edit on the
   generated wrapper or an XDC synth-time directive, mechanism not yet
   decided) or 4 (toggle `Pool_hls`'s IP synth mode).   **TESTED 2026-10-04: workaround 2 (`KEEP_HIERARCHY`/`DONT_TOUCH` on
   `Pool_hls`) also FAILED.** Added `(* keep_hierarchy = "yes" *)` and
   `(* dont_touch = "yes" *)` directly on the generated
   `finn_design_Pool_hls_0_0` wrapper module declaration (the BD IP wrapper
   instantiating `Pool_hls_0`), then reran OOC synth directly against the
   same stitched project (bypassing a full pipeline rerun). Result:
   `finn_design_Pool_hls_0_0` now gets its own explicit "Retiming module
   ... done" line (proof the attribute took effect -- it wasn't silently
   flattened away), and the crash fires IMMEDIATELY after that line, at
   the transition to whatever Vivado processes next. This matches (not
   contradicts) the original finding that `Pool_hls_0` itself always
   synthesizes/retimes cleanly -- 3/3 structural workarounds (BRAM, global
   flatten, per-instance keep-hierarchy) now agree the crash is NOT in
   processing `Pool_hls_0` itself, but in whatever comes immediately after
   it in Vivado's internal sequence, regardless of which synthesis
   strategy is used. Remaining untried: workaround 4 (toggle `Pool_hls`'s
   IP synth mode OOC vs Global -- would make `Pool_hls_0` arrive at this
   stage already-frozen rather than actively retimed, a materially
   different mechanism than 1-3) and workaround 5 (`vivado -stack 2000`).
   **TESTED 2026-10-04: workaround 5 (`vivado -stack 2000`) also FAILED.**
   Added `-stack 2000` to the `vivado` invocation in `vivadocompile.sh`,
   reverted after test. Identical crash, identical position (right after
   `Pool_hls_0` finishes retiming). Rules out a plain Tcl interpreter
   stack overflow as the cause.
   **INVESTIGATION PAUSED 2026-10-04**: 4 of 5 AMD-documented workarounds
   tested (`-max_bram 0`, `-flatten_hierarchy none`, `KEEP_HIERARCHY`/
   `DONT_TOUCH` on `Pool_hls`, `-stack 2000`) all fail identically. Only
   untried option is #4 (pre-synthesizing `Pool_hls_0` into its own
   checkpoint and `read_checkpoint`-ing it into the main project), which
   is materially more invasive to wire up since this pipeline uses flat
   Verilog sources rather than managed IP (no simple property toggle
   exists). Not attempted. Current state: the InferPool route for the
   initial block remains blocked by this crash; the stock MaxPool +
   `thrpre` reorder route is the working fallback.
   **ISOLATION TEST 2026-10-04: standalone OOC synth of `Pool_hls_0` alone
   SUCCEEDS.** Built a minimal single-IP Vivado project (just the 4
   generated `Pool_hls_0*.v` files, no other stitched-design sources)
   with attributes exactly matching the real failing node (signed INT4,
   PE=1, Channels=1, KernelSize=[2,2], same part/clock) and `-retiming`
   enabled to match `vivadocompile.tcl`. Result: `synth_design` completes
   with 0 errors, 0 critical warnings. This rules out `Pool_hls_0`'s own
   generated RTL (including its retiming behavior) as the trigger -- the
   crash only occurs in the full stitched-design context, confirming
   this is a cross-module/cross-boundary interaction (AMD KB 946862's
   category), not a defect in this IP's own netlist.
   **ISOLATION TEST 2026-10-04 (extended): 3-IP combo also SUCCEEDS.**
   Extended the isolation to the real upstream chain feeding `Pool_hls_0`:
   `ConvolutionInputGenerator_rtl_1` (the SWG instance whose *output tensor*
   feeds it -- node-name vs. tensor-name are swapped between the two SWG
   instances in this graph, traced via the real producer/consumer tensor
   names, not node index) -> `StreamingFIFO_rtl_7` (depth-2 `Q_srl`) ->
   `Pool_hls_0`. Built a flat top wrapper directly instantiating all 3 IPs'
   real generated sources (no Vivado IP Integrator/BD, no `.xci` packaging)
   with the same `-retiming` flag and clock constraint. Result: 0 errors
   again. This rules out the SWG+FIFO+Pool local neighborhood as the
   trigger too -- the crash needs more of the full stitched design than
   just these 3 IPs. Leading untested hypothesis: production builds go
   through Vivado's IP Integrator/Block Design flow (packaged `.xci` +
   auto-generated `finn_design_wrapper.v`), which this flat-instantiation
   test does not replicate -- that codegen difference hasn't been isolated
   as a variable yet.
   **ISOLATION TEST 2026-10-04 (real FINN pipeline): hypothesis REFUTED.**
   Wrote a standalone script, `thresh_pool_bd_isolation.py` (kept in repo,
   a reusable minimal isolation tool, not a throwaway), that hand-builds a
   2-op ONNX graph (`MultiThreshold` -> `MaxPool`, same structural params:
   channels=1, 256x256->128x128, kernel/stride 2x2) and runs it through the
   REAL FINN pipeline end to end -- real `to_hw.InferPool`/
   `InferThresholdingLayer`/`InferConvInpGen`, real `step_specialize_layers`/
   `step_hw_codegen`/`step_hw_ipgen`/`step_set_fifo_depths`, and critically
   the REAL `CreateStitchedIP` (genuine Vivado IP-Integrator/Block-Design
   flow with `.xci` packaging and an auto-generated `finn_design_wrapper.v`
   -- NOT a hand-written flat top) and the REAL, unmodified
   `SynthOutOfContext` (same `vivadocompile.tcl`/`.sh` production uses).
   Resulting HW graph was confirmed identical in shape to the flat test:
   `Thresholding_rtl_0 -> ConvolutionInputGenerator_rtl_0 -> Pool_hls_0`.
   Result: **0 errors**, WNS=+6.933ns (comfortably meets the 10ns clock),
   fmax=326MHz. This REFUTES the "BD/stitching-mechanism-specific" theory
   floated after the flat-combo test -- going through the real IP-Integrator
   flow with only 3 real IPs still doesn't crash. The remaining, now
   better-supported theory: the crash is a scale/hierarchy-flattening
   interaction that only appears with the FULL ~20-IP design, not a defect
   tied to any small IP subset or to the stitching mechanism itself. Next
   step if resumed: grow `thresh_pool_bd_isolation.py`'s hand-built ONNX
   graph incrementally (FMPadding, DuplicateStreams, the MVAU branch,
   Concat, more Thresholds) toward the full init-block topology to find the
   point where it starts crashing.
   **ISOLATION TEST 2026-10-05 (dup+concat skip connection): STILL PASSES.**
   Grew the previous test into a new script, `thresh_pool_dup_concat_bd_
   isolation.py` (kept in repo): forks the single `Thresholding` output into
   2 consumers (genuine fork -> `to_hw.InferDuplicateStreamsLayer`
   auto-inserts `DuplicateStreams_hls`, not manually placed), each consumer
   is an independent `SWG -> Pool_hls` branch (same 2x2/stride2 params),
   rejoined by a channel-axis `Concat` -- mirroring the real initial
   block's "pool branch + conv branch, concatenated on channel axis"
   topology. Lowering Concat requires axis=-1 (HW `StreamingConcat` only
   supports channel-last), so reused `hardware/finn_enet_build.py`'s
   `_move_transpose_past_concat` trick (inlined self-contained, no
   cross-import) to push both branches' trailing Transposes past the
   axis=1 Concat first. Confirmed final HW graph via the real pipeline:
   `Thresholding_rtl_0 -> DuplicateStreams_hls_0 -> {ConvolutionInputGenerator_
   rtl_0 -> Pool_hls_0, ConvolutionInputGenerator_rtl_1 -> Pool_hls_1} ->
   StreamingConcat_hls_0`. Run through the same real `CreateStitchedIP` +
   real `SynthOutOfContext`. Result: **0 errors again**. LUT=1169,
   LUTRAM=196, FF=781, Carry=59, DSP=0, BRAM=0, WNS=+6.77ns, fmax=309.6MHz.
   Even with 7 real IPs including TWO `Pool_hls` instances, a
   `DuplicateStreams`, and a `StreamingConcat`, the crash still doesn't
   reproduce -- further strengthens the scale/hierarchy-flattening
   hypothesis over any small-subgraph explanation. Next step if resumed:
   keep growing this script (FMPadding, the MVAU/conv branch, more
   Thresholds) toward the full init-block topology, or switch to top-down
   binary-search node removal from the full failing probe case instead of
   bottom-up addition.
## New finding this round: `noconv` and `init`/`fnl` throughput shortfalls

Not previously flagged as a problem in the gotchas file — these all complete
rtlsim and OOC cleanly, they just **miss their throughput target**, which is
a folding/parallelization gap rather than a crash:

- `up_cin32_cout16_in32_int{4,6,8}_noconv`: ~20.3-20.5 cyc/px vs 18.0 target
  (~14% over budget). **ROOT-CAUSED 2026-10-04** via the 6-case `SET=upd`
  diagnostic batch (handoff §16): the FIFOs immediately around
  `UpsampleNearestNeighbour` are undersized (stock 8 words each side vs. the
  ~40/91 words needed for `mvau_p` to run ahead while the upsampler re-emits
  a row). Evidence: `_fupnn` (resize just those two FIFOs) alone recovers to
  PASS (16.00 cyc/px), `_fall` (fupnn+fjoin) gives the identical 16.00 -- so
  `fjoin` (the `Thr_e->Add` join FIFO resize) contributes nothing on its own
  (`_fjoin` alone stays SLOW at 20.53, unchanged from baseline). Raising PE
  (`_pe2`/`_pe4`) also independently recovers it (overprovisioned compute
  absorbs the same stall differently), but the FIFO-sizing bug in
  `up_bottleneck_sim.py`'s folding-config generator for the `noconv` path is
  the actual root cause, not a join-chain PE-slack issue. Fix: widen the
  `dwc->upnn`/`upnn->...` FIFOs in the `noconv` folding config generator to
  match `_fupnn`'s values (40/91 words) instead of raising PE/SIMD.
- `init_cin1_cout4_in256_int{4,6,8}` (non-pool route): 15.78 cyc/px vs 5.0
  target (>3x over) — the pool-route variant (`_int4_pool`) comes much closer
  (5.01 vs 4.2) before hitting the separate Vivado-crash issue above.
  **PARTIALLY INVESTIGATED 2026-10-04**: landed the `MoveMaxPoolPastMultiThreshold`
  graph reorder (threshold upstream of the maxpool, matching the real
  FINN-lowering order) and re-ran as `SET=intpre` (`_thrpre` twins, same
  ONNX, reordered graph). Result: 13.04 cyc/px for all of int4/int6/int8 —
  better than 15.78 (~17% faster) but still SLOW, still >2.6x over target,
  and identical across all three bit widths (same signature as `fnl_nobias`
  below — points to a structural bottleneck independent of folding/bit
  width, not yet identified). The reorder was necessary groundwork (it's
  what lets the pool branch's `MaxPool` become eligible for `MakeMaxPoolNHWC`
  at all) but is not sufficient on its own to close the gap.
- `fnl_cin4_cout5_in128_int{4,6,8}_nobias`: 763.95 cyc/px vs 1.1 target
  (~700x over) — this looks like a real folding/parallelization gap in the
  final-deconv probe's config, not noise. The `_bias` variant was only ever
  run once (`NO-RTLSIM`) and never retried after whatever stopped it.

None of these three have been root-caused yet — they're folding/config gaps
in `make_up_folding_configs.py` / `make_int_folding_configs.py` /
`make_fnl_folding_configs.py` (or the underlying `MILP/analytical` cost
model), not confirmed hardware/toolchain bugs like items 1-3 above. Next step
for whoever picks this up: diff each case's `stages.folding` (landed PE/SIMD)
against `predicted.nodes` (see `probe_result.json`) to find which node(s)
account for the gap.

## Open / not yet done

- Apply the `noconv` FIFO-sizing fix (widen `dwc->upnn`/`upnn->...` to
  40/91 words) in `up_bottleneck_sim.py`'s folding generator and re-verify
  `up_cin32_cout16_in32_int{4,6,8}_noconv` lands PASS without needing a PE bump.
- Root-cause the remaining two throughput shortfalls (`init` non-pool,
  `fnl_nobias`) -- the `noconv` one above is now resolved. `init` non-pool
  improved 15.78->13.04 via the thrpre reorder but is still unexplained;
  next step is diffing `stages.folding` vs `predicted.nodes` per-node to find
  which HW node accounts for the remaining ~2.6x gap.
- Retry/investigate `fnl_cin4_cout5_in128_int4_bias` (only one `NO-RTLSIM`
  attempt on record, never followed up).
- RTL MVU Verilog bug (item 2): try a different PE/SIMD folding, or escalate
  upstream to FINN.
- `init_cin1_cout4_in256_int4_pool` Vivado crash (item 3): matches AMD's
  documented "Cross Boundary Optimization" crash class (KB 946862), not an
  opaque one-off. 4 of 5 AMD-documented workarounds TESTED 2026-10-04 and
  ALL FAILED: `-max_bram 0`, `-flatten_hierarchy none`, `KEEP_HIERARCHY`/
  `DONT_TOUCH` on `Pool_hls`, `vivado -stack 2000` -- all produce the
  identical crash at the identical relative position (immediately after
  `Pool_hls_0` finishes its own processing, whatever synthesis strategy is
  used). Only remaining untried option (#4, pre-synthesizing `Pool_hls_0`
  into its own checkpoint via `read_checkpoint`) requires a materially more
  invasive sub-flow since this pipeline uses flat Verilog, not managed IP.
  Investigation paused here. Current working fallback: stock MaxPool +
  `thrpre` reorder (already functional) for the initial block.
  **ISOLATION TEST 2026-10-04**: standalone single-IP OOC synth of
  `Pool_hls_0` alone (exact attribute match incl. retiming) SUCCEEDS with
  0 errors -- confirms the crash is a cross-boundary interaction with the
  rest of the stitched design, not a defect in `Pool_hls_0` itself.
  **EXTENDED 2026-10-04**: a flat 3-IP combo (`ConvolutionInputGenerator_rtl_1`
  -> `StreamingFIFO_rtl_7` -> `Pool_hls_0`, the real upstream neighborhood,
  exact attributes + retiming) also SUCCEEDS with 0 errors -- the crash
  needs more design context than just this local neighborhood. Untested
  next step: replicate via Vivado's actual IP Integrator/Block Design flow
  (packaged `.xci` + auto-generated `finn_design_wrapper.v`) instead of a
  hand-written flat top, since that's what production builds actually use
  and this flat test doesn't exercise that codegen path.
  **REAL-FINN-PIPELINE TEST 2026-10-04**: `thresh_pool_bd_isolation.py` ran
  the same 3-node chain through the REAL FINN build pipeline, including the
  genuine `CreateStitchedIP` (Vivado IP-Integrator/BD, `.xci` packaging) and
  the real unmodified OOC synth script -- still 0 errors. This REFUTES the
  BD/stitching-mechanism hypothesis; the crash needs the full ~20-IP design
  scale, not just the real invocation mechanism with few IPs.
  **DUP+CONCAT TEST 2026-10-05**: `thresh_pool_dup_concat_bd_isolation.py`
  grew the above to a skip-connection shape (`Thresholding` ->
  `DuplicateStreams_hls` -> 2x `[SWG -> Pool_hls]` -> `StreamingConcat_hls`,
  7 real IPs incl. two `Pool_hls` instances), same real `CreateStitchedIP`
  + `SynthOutOfContext` -- still 0 errors (LUT=1169, FF=781, WNS=+6.77ns).
  Further strengthens the scale/hierarchy-flattening hypothesis.
