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
  (~14% over budget). The `conv` variant (same Cin/Cout) hits target fine, so
  this is specific to the `noconv` topology's own folding, not a shared bug.
- `init_cin1_cout4_in256_int{4,6,8}` (non-pool route): 15.78 cyc/px vs 5.0
  target (>3x over) — the pool-route variant (`_int4_pool`) comes much closer
  (5.01 vs 4.2) before hitting the separate Vivado-crash issue above.
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

- Root-cause the three throughput shortfalls above.
- Retry/investigate `fnl_cin4_cout5_in128_int4_bias` (only one `NO-RTLSIM`
  attempt on record, never followed up).
- RTL MVU Verilog bug (item 2): try a different PE/SIMD folding, or escalate
  upstream to FINN.
- `init_cin1_cout4_in256_int4_pool` Vivado crash (item 3): clear `.Xilinx`
  cache dir, try different folding, or isolate the InferPool route with a
  minimal repro.
