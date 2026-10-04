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
3. **`init_cin1_cout4_in256_int4_pool`** — OOC synth hits a deterministic
   Vivado-internal Tcl-stack crash (`TclStackFree: incorrect freePtr`) during
   Technology Mapping. Not fixed; not retried since.

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
- `init_cin1_cout4_in256_int4_pool` Vivado crash (item 3): clear `.Xilinx`
  cache dir, try different folding, or isolate the InferPool route with a
  minimal repro.
