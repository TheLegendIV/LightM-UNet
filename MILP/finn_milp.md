# finn_milp.py — formulation, rationale, and history

Companion reference for `finn_milp.py`. The `.py` keeps only short comments;
this file holds *why* each piece exists, the evidence behind it, what is known
to be missing, and the output schema. **Agents: update this file whenever you
change the solver's behavior, inputs or outputs.** Cost formulas and their
calibration are documented separately in `finn_cost_model.md`.

## Pipeline position

```
layer_sensitivity.py  →  finn_milp.py  →  expand_layer_bits.py  →  calibrate_*.py / QAT trainer  →  FINN build
   (HAWQ traces)          (bits + folding)   (per-layer → per-site bits)
```

Module dependencies: `finn_cost_model.py` (cost formulas), `layer_topology.py`
(predecessor map + dataflow graph), `block_utils.py` (block membership),
`expand_layer_bits.py` (`resolve_act_sources` — the deployed bit rule for
`out_act`), `milp_outputs.py` (pruning report + ONNX export), one
`configs/config_*.py`.

## Formulation

Per individual layer (every Conv2d / ConvTranspose2d / MaxPool2d), plus every
extra dataflow node (see "Dataflow graph"):

**Variables**
- `y[layer, w, a]` — binary, one-hot per layer: the (weight_bits, act_bits)
  pair. Sensitivity attaches here.
- `z[layer, pe, simd, thr_ram_style, variant, w, a]` — binary, one per
  (fold, variant, bits) combination. Cycles / LUT / BRAM / DSP / URAM attach
  here, evaluated by `finn_cost_model.layer_cost_pe_simd` at that exact point.
- `z[node, pe, simd, ram_style, variant, w, bits]` — the same, for an extra
  node, over that node's own legal options (`extra_node_options`).
- `max_downstream_rate[node]` — continuous, only with `--dsr-ratio`.

**Objective**
```
minimize  (1/n_layers) * Σ y · sens_norm
```
Accuracy-only: `sens_norm` is a global min-max normalization of raw HAWQ
sensitivity to [0, 1]. The folding variables `z` carry **no objective
weight at all** — every fold/variant/ram_style choice that satisfies the
hard constraints below is equally optimal, so the solved `pe`/`simd` is a
free (solver-arbitrary) pick among the feasible set, not a speed decision.
There used to be an `alpha` dial trading this off against a mean-cycles
term (`alpha=1.0` = this objective, `alpha=0.0` = pure cycles); it was
removed 2026-09-28 (see "History") because every real run used `alpha=1.0`
anyway — this file's own worked example at `--max-latency-ms` below was
already describing that fixed point. If a genuinely speed-weighted solve is
ever needed again, reintroduce the dial rather than repurposing this one.

**Constraints**
- One `(w, a)` per layer; `--pin-bits-file` pins it (TEST-ONLY).
- Link: `Σ_folds z[layer, fold, w, a] == y[layer, w, a]` — exactly one fold
  at the chosen bits.
- Extra nodes: exactly one option each; threshold bits with sources tied to
  `max(sources)` (below).
- Optional rate coherence (`--pbi-ratio` / `--dsr-ratio`, below).
- Hard budgets, always enforced (not soft penalties):
  `Σ z·LUT ≤ f_lut·230,400`, `Σ z·BRAM18 ≤ f_bram·624`,
  `Σ z·DSP ≤ f_dsp·1,728`, `Σ z·URAM ≤ f_uram·96` (xczu7ev-ffvc1156-2-e).
- Optional throughput (`--target-fps`): `cycles[n] ≤ clock_mhz·1e6 / target_fps` for
  **every** hardware node. A streaming pipeline delivers one frame per slowest-node
  frame time, so this cap is exactly the FPS target (e.g. 250 FPS @ 100 MHz → every
  node ≤ 400,000 cycles). Nodes whose fully-folded minimum already exceeds the cap
  are reported in `_diagnostics.throughput_floor_violations`. The result reports
  the achieved `bottleneck_node` / `bottleneck_cycles`, and `summary.csv` the FPS.
- Optional latency: `Σ z·cycles ≤ max_latency_ms/1000 · clock_mhz · 1e6`.
  Note this is a **sum** of per-node frame cycles (a latency proxy that
  over-counts parallel branches), not a throughput bound.

With `--max-latency-ms` (and/or `--target-fps`), the solve is an
epsilon-constraint formulation: best accuracy proxy subject to the latency/
throughput cap(s) — the convention for the deployed S12 runs, and now the
solver's only mode (see "History").

## Sensitivity

Source: `layer_sensitivity.py`'s `layer_sensitivity_<config>.json`, keyed by
the same per-layer names `trace_layer_geometry` produces.

**Predecessor-corrected act sensitivity.** `layer_sensitivity.py` measures a
layer from a forward hook on its own module — its OWN OUTPUT. But the cost
model uses a layer's `act_bits` as its OWN INPUT stream's bit-width (e.g.
BRAM_swu's `ceil(C*A/36)` line-buffer term sizes the buffer receiving the
incoming feature map). So the act term for layer L reads
`max(sensitivity[pred]["sensitivity_a"][a] for pred in real predecessors)`,
using `layer_topology.compute_predecessor_map` (conv-only). MAX, not mean,
because a residual join can have two real predecessors and a wire is only as
safe to compress as its most sensitive contributor. No predecessor (first
layer) or tracing failure → falls back to the layer's own sensitivity.

**MaxPool2d** has no weights and is never HAWQ-measured → raw sensitivity 0.0
for every (w, a). Because normalization is one global affine map, a constant
per-layer value adds a constant to the objective but never changes which (w, a)
that layer picks — MaxPool bits are chosen purely on cost.

**Zero-sensitivity layers** (`trace_w` and `trace_a` both ~0, flagged by
`layer_sensitivity.py`'s detector, stored as `_zero_sensitivity_layers`) are
warned about and reported in `summary.csv` and `pruning_*.json`. A dead layer
gets cheap bits (correctly), but its near-zero calibrated quantizer scale can
underflow to exactly 0.0 under fp16 deployment (the `regular5.0` incident) —
pruning is the better fix.

## Dataflow graph (extra nodes)

`layer_topology.compute_dataflow_graph` returns the real FINN dataflow graph —
every hardware node FINN v0.10.1 lowers the model to, including the ones with
no conv/pool module — plus a `kind` per virtual node. `insert_skip_pads` then
adds the downsampling pad-MVAU (it needs shapes). Each extra node gets its own
fold options (exactly what the FINN op supports), priced in every budget, the
latency term, `max_cycles` and the rate constraints.

**Source of truth.** FINN **v0.10.1** code (the version the S12 builds used —
its op attributes match the built graphs exactly; the local checkout is
v1.0.0-alpha, which already removed `AddStreams`) and the Brevitas 0.12.1
model code — not exported ONNX graphs.

| kind | FINN op | folding (v0.10.1) | cycles | resources | bits |
|---|---|---|---|---|---|
| `input_quant` | Thresholding_rtl | PE \| C | H·W·⌈C/PE⌉ | threshold fit | `max(sources)` = deployed site (`initial.conv`) |
| `concat` | StreamingConcat_hls | none (all channels/cycle) | H·W | 0 (unknown) | — |
| `act` (InitialBlock) | Thresholding_rtl | PE \| C | H·W·⌈C/PE⌉ | threshold fit | `max(initial.conv, initial.pool)` |
| `skip_quant` | Thresholding_rtl | PE \| C | H·W·⌈C/PE⌉ | threshold fit | fixed 8 |
| `pad_mvau` | MVAU_rtl (1×1, cin→cout) | PE, SIMD | MVAU formula | MVAU formula | fixed W2 / A8 (provisional) |
| `add` | AddStreams_hls | PE \| C | H·W·⌈C/PE⌉ | 131 LUT/PE (provisional) | — |
| `residual_add` | Thresholding_rtl | PE \| C | H·W·⌈C/PE⌉ | threshold fit | fixed 8 |
| `out_act` | Thresholding_rtl | PE \| C | H·W·⌈C/PE⌉ | threshold fit | `max(sources)` = deployed rule |
| `dup` | DuplicateStreams_hls | PE \| C | H·W·⌈C/PE⌉ | 115 LUT/PE (provisional) | — |
| `upsample` | UpsampleNearestNeighbour_hls | none (all channels/cycle) | H_out·W_out | 0 (unknown) | — |

Plus, inside every padded conv's own cost: **FMPadding** (and FMPadding_Pixel +
border padding for ConvTranspose), cycles = padded H·W·⌈C/SIMD⌉ with SIMD tied
to the conv's SWU (no data-width converter between them); the conv's cycles are
`max(MVAU, SWU, FMPadding)`. FINN has no resource estimator for it.

Notes and evidence:
- **Residual-join thresholds** (`skip_quant`, `residual_add`, `out_act`):
  real-build evidence and per-node synthesis numbers in `finn_cost_model.md`
  "Residual-join thresholds". The main operand's requant is absorbed in
  `expand`'s own threshold (already priced as `expand.0`'s `thr_*`).
- **Why `skip_quant`/`residual_add` are 8-bit.** Brevitas 0.12.1's
  `QuantEltwiseAdd` only forwards `input_*`/`output_*`-prefixed kwargs to its
  quantizers, and `output_quant` defaults to `Int8ActPerTensorFloat`. Verified
  at runtime: `QuantEltwiseAdd(bit_width=4, input_quant=Int8ActPerTensorFloat)`
  → 8-bit input quant, 8-bit output quant. **Consequence:** every
  `act_bits["residual_add"]` HAWQ / the ILP / `expand_layer_bits.py` assigns is
  silently ignored by `LayerQuantENet`/`QuantENet` (they'd need
  `input_bit_width=`/`output_bit_width=` to take effect). `QuantIdentity` and
  `QuantReLU` do honor `bit_width` (verified), so `input_quant`, `act` and
  `out_act` follow their site bits.
- **Bit ties.** Sources come from `expand_layer_bits.resolve_act_sources`, so
  the ILP prices the deployed bits. Encoded exactly with cumulative indicators:
  for every bit level b, `[bits_V ≥ b] = OR_s [bits_s ≥ b]`, linearized as `≥`
  each source and `≤` their sum — linear, no extra binaries.
- **Forks.** FINN inserts one DuplicateStreams per tensor with ≥2 consumers;
  the graph does the same (28 in S12). A fork's `dup` belongs to the block
  that consumes it, so pruning that block frees it.
- **Shape queries are not data.** `F.interpolate(x, size=other.shape)` reads
  another tensor's shape; the walk ignores `.shape` / `.size()` so it creates no
  fake edge or fork.
- **MaxPool is not foldable** (2D): v0.10.1's `StreamingMaxPool_Precision`
  template takes no PE — `PE` only reaches the 1D variant — and its folded shape
  carries all channels in one word; cycles = H·W·(1+1/k²). SetFolding does not
  touch it.
- **SetFolding vs these knobs.** v0.10.1 `SetFolding` folds AddStreams,
  DuplicateStreams and thresholds (PE) and FMPadding_hls / FMPadding_Pixel
  (SIMD); not FMPadding_rtl, StreamingMaxPool, Upsample or Concat. The real
  builds applied a HAWQ folding config instead and left every AddStreams /
  DuplicateStreams at PE=1 — up to 262k cycles per node at 128×128×16, i.e. in
  the range of the slowest convs. The ILP's PE choice for these nodes must be
  written into the folding config to take effect.
- **Provisional resources.** FINN v0.10.1 prices all stream nodes at 0 (no
  estimator overrides). AddStreams/DuplicateStreams LUT are Vitis HLS csynth
  estimates at PE=1, 8-bit, scaled linearly in PE (unverified). Concat,
  Upsample and FMPadding: no data → 0. Pad-MVAU bits (W2/A8) are a guess: its
  weights are a constant identity/zero matrix whose FINN datatype wasn't
  checked. Calibrate from per-node Vivado hierarchy reports
  (`hier_partition_N.rpt`, no longer present locally).
- **Not modeled:** FIFOs, data-width converters, the extra INT8 threshold seen
  on the InitialBlock pool path in the built graph (no Brevitas quantizer
  explains it).
- Real 8-way build: the residual-join thresholds alone were 68 nodes =
  4,592 LUT and ~247 BRAM18-eq unpriced before 2026-09-26 — older solves
  under-report BRAM badly (the deployed `bram20` plan reported 124 BRAM18).

## Rate coherence: `--pbi-ratio RATIO` (join balance) and `--dsr-ratio RATIO` (chain coherence)

Two independent constraint families on the dataflow graph (so threshold
nodes and ordinary residual joins are included), each gated by its own flag.
**2026-09-29:** split back into two flags from a single merged
`--optimize-downstream-rate` that drove both at once (itself a 2026-09-26
merge of two separately-named flags — see History) — a user-facing sweep
needed to hold one mechanism fixed while varying the other, which a single
shared ratio can't express. `pbi_ratio`/`dsr_ratio` (the internal parameter
names) each default to `None` (off); either, both, or neither may be set.

1. **`--pbi-ratio` — join balance (parallel branch imbalance).** For every
   simple 2-branch diamond (fork → {branch A, branch B} → join,
   `find_fork_join_diamonds` in `layer_topology.py`), both ways:
   `sum(cycles over branch A) ≤ ratio · sum(cycles over branch B)`. A
   "branch" is every node from the join's direct predecessor back to (not
   including) the nearest fork — the WHOLE path, not just that one direct
   predecessor. Symmetric — siblings have no inherent order. A diamond is
   exempt only if NEITHER branch has any real folding freedom at all
   (every node on that branch is fixed-cycle — MaxPool, concat, upsample —
   so it can't respond to a rebalancing constraint).
   (**Corrected 2026-09-29** — the original implementation compared only
   the single node immediately before a join, e.g. `skip_quant` vs.
   `expand.0` at a residual add, ignoring every node earlier on the longer
   branch (`reduce`, `conv`). This misses a branch that's individually
   rate-compliant node-by-node but simply has more pipeline stages than its
   sibling — confirmed empirically: forcing the old single-node check to an
   exact 1.0 ratio left the real cumulative diamond ratio completely
   unchanged, 1.8x, on the `nearest_upsample` architecture, because the
   3-stage main branch's earlier nodes were never compared against
   anything. The diagnostic-only `milp_outputs.compute_branch_imbalance_report`
   already computed this same cumulative-sum quantity — this promotes it
   from a diagnostic to the actual enforced constraint, and both now share
   `find_fork_join_diamonds` so they can never diverge on what counts as a
   diamond again.)
2. **`--dsr-ratio` — chain coherence (downstream rate).** `rate[D] ≤ ratio ·
   rate[L]` for every descendant D of an ancestor L, with
   `rate = cycles / (C·H·W of the node's own output)` so nodes are
   comparable across resolution changes. Only the "producer outruns
   consumer" direction is constrained — a slow producer only starves a FIFO
   (throughput loss), a fast one fills it (depth risk): D's rate (however
   slow) is capped relative to L's, so L can't be more than `ratio`× faster,
   per output element, than anything reachable downstream of it.
   (**Corrected 2026-09-29** — an earlier revision of this doc stated the
   inequality with L and D on the opposite sides, `rate[L] ≤ ratio ·
   rate[D]`; re-tracing it against the toy 3-node example below shows that
   direction is vacuous for the exact compounding-mismatch case this feature
   exists to catch, while `rate[D] ≤ ratio · rate[L]` is the one the
   `max_downstream_rate` encoding below actually enforces and that correctly
   rejects that example.)

Why all descendants, not just neighbors: FIFO depth is driven by the
total-cycles-per-frame mismatch between producer and consumer, and
backpressure from a slow node several hops downstream propagates upstream
through every FIFO. A fork needs no special case — independent constraints
from each branch B back to the fork F are exactly `rate[B] ≤ ratio ·
rate[F]`, applied per branch.

Encoding: all-pairs is O(n²) (≈14k dense constraints with threshold nodes), so
it uses `max_downstream_rate[L] ≥ rate[c]` and `≥ max_downstream_rate[c]` for
each child c, then `max_downstream_rate[L] ≤ ratio · rate[L]`. Exact and
O(edges). `max_downstream_rate[L]` is the slowest (bottleneck) rate anywhere
in L's downstream subtree, bounded **below** by real rates, so it cannot be
deflated to cheat the outer `≤`.

**2026-09-26 correction.** The previous encoding tracked `min_downstream_rate`
(the *fastest* descendant, bounded above) and constrained
`rate[L] ≤ ratio · min_downstream_rate[L]`. Traced against a toy 3-node chain
L→M→D with `rate = {L:1, M:1, D:10}` (L, M fast, D a slow bottleneck several
hops downstream — exactly the compounding-mismatch case this feature exists
for): that constraint was fully satisfied and non-binding, because L and M
were only ever compared against the fastest thing downstream of them (each
other), never against the true bottleneck D. The reverse scenario
(`rate = {L:10, M:1, D:1}`, a slow producer) was correctly rejected — exactly
backwards from "producer outruns consumer is the constrained direction" above.
Root cause: when the vacuous-inflation bug (next paragraph) was fixed by
switching the auxiliary's bound direction, the tracked extremum (min vs. max)
flipped with it without re-verifying the outer inequality still matched the
documented intent. Fixed by swapping to `max_downstream_rate`, bounded below
instead of above — same anti-inflation principle, mirrored: a MAX-type
auxiliary sitting on the restricted (left) side of a `≤` must be bounded from
below only, so it can't be pushed down to cheat the constraint (the mirror
image of the MIN-type fix, which sat on the generous/right side and had to be
bounded from above only).

(An even earlier draft used a running *max* bounded only from below while
ALSO sitting on the generous side of its own outer constraint — that could be
inflated for free to satisfy it vacuously; a different bug from the one
above, already fixed before this session's chain-coherence work started.)

Neither family prices the FIFO itself; they only exclude badly skewed plans.

History: `--max-join-imbalance-ratio` (join-only, separate flag) and an
unnamed chain-coherence flag were folded into one shared
`--optimize-downstream-rate` on 2026-09-26, then **split back into
`--pbi-ratio` and `--dsr-ratio` on 2026-09-29** so the two mechanisms can be
swept independently again (see top of this section). Before the dataflow
graph existed, the conv-only predecessor map's one-branch-point cap hid
every ordinary chained RegularBottleneck join — only UpsamplingBottleneck
joins were visible.

## Constants

- `XCZU7EV = {LUT: 230,400, BRAM_18K: 624, DSP: 1,728, URAM: 96}` —
  xczu7ev-ffvc1156-2-e. DSP matches `hardware/results.csv`'s DSP_pct column;
  URAM=96 URAM288 blocks, confirmed against the real S12-dense-warmstart build
  (URAM=2/96).
- `CANDIDATE_BITS = (2, 4, 6, 8, 16)` — usually overridden with
  `--candidate-bits` (e.g. `4,6,8`). Every extra (w, a) pair multiplies every
  layer's fold count.
- `INPUT_HW = (512, 512)` — the real nnU-Net patch size.
- `RESIDUAL_QUANT_BITS = 8` — see Residual-join thresholds.

### `RAM_STYLES` and URAM (history)

`RAM_STYLES = (block, distributed)` drives the **standalone Thresholding
node's memory only** (`thr_ram_style`). MVAU weight memory is hard-fixed to
block, SWU line buffer to distributed.

The axis went through several states on 2026-09-17/18, each block-vs-URAM:
MVAU weight memory, then SWU line buffer, then Thresholding memory. Each had
no working URAM path on this device:
- An uncapped `alpha=0.25` solve picked `wm_uram18=205`; the real 8-way build
  landed at URAM=2/96 — real hardware never realized "ultra" for MVAU weights.
- 48/48 real SWU nodes (both datasets) used `ram_style="distributed"`
  regardless of the request.
- A real Vivado synthesis of `Thresholding_rtl` with `ram_style="ultra"`
  **fails**: URAM288 cannot be ROM (no INIT-file initialization), and the
  threshold table is a compile-time constant.

So URAM is structurally 0 everywhere; `XCZU7EV["URAM"]` /
`--hard-uram-fraction` are kept only for provenance.

The axis now drives block vs **distributed** (LUTRAM) for thresholds — real:
`thresholding.sv`'s `RAM_STYLE` localparam has a genuine distributed branch,
forceable per node via `depth_trigger_bram`. Never exercised in a real build
(0/240 real Thresholding nodes showed LUTRAM; Vivado's "auto" always chose
BRAM). The LUTRAM cost is derived (see `finn_cost_model.md`) and predicts
distributed is a bad deal at numSteps=255 (~840 LUT vs ~3 BRAM18) but can win
at small numSteps — a per-layer tradeoff the ILP evaluates. Watch for solves
that push many thresholds to distributed under tight BRAM (a 2026-09-26
BRAM-50% solve put 56.8k LUT into thresholds this way).

FIFOs are not a modeled resource at all.

## Resource variants

- `rtl_dsp_noact1` — default and only variant unless `--allow-lut-mult`:
  MVAU_rtl, DSP multipliers, standalone Thresholding (`noActivation=1`).
  Depthwise layers resolve to HLS inside the cost model (VVAU_rtl needs a
  Versal DSP58), which matches real behavior.
- `hls_lut_noact0` (`--allow-lut-mult`, added 2026-09-17) — HLS backend, LUT
  multipliers, fused activation. Lets the ILP spend spare LUT instead of DSP
  per layer. Its fused-threshold LUT term is a FINN-source transcription, not
  calibrated on real hardware. `--force-dsp` deliberately does not affect it.
- Not in the curated set: `rtl_dsp_noact0` (illegal — MVAU_rtl requires
  `noActivation=1`), `hls_dsp_noact1` (dominated by `rtl_dsp_noact1`).
- `hls_dsp_noact0` — **placeholder, never eligible**. `_variant_cost_kwargs`
  handles it, but before enabling it, hard-restrict its folds to PE ≤ SIMD:
  on 89 real MVAU_hls nodes of exactly this kind (resType=dsp, fused
  threshold; `hardware/datasets/mvau_lut_calibration_dataset.csv`,
  `hardware/mvau_lut_correlation_report.txt`) the structural LUT formula gets
  R²=0.009, PE alone correlates 0.69, SIMD is negatively correlated (−0.41),
  and PE>SIMD rows are 36% of rows but 70.7% of real LUT.

`_calibration_force_dsp`: `rtl_dsp_noact1` always gets the RTL LUT derate
inside `conv_cost_pe_simd` (RTL is DSP-only), so `calibrated_lut` must always
bypass its avg_bits table for it — not only when `--force-dsp` is set.
Every run so far passed `--force-dsp`, which masked this.

## CLI flags

| flag | notes |
|---|---|
| `--config` | `MILP/configs/config_*.py`, injected into module globals. No default config. |
| `--sensitivity-file` | `layer_sensitivity_*.json`. Must cover every conv layer (MaxPool may be absent). |
| `--candidate-bits` | e.g. `4,6,8`. |
| `--hard-{lut,bram,dsp,uram}-fraction` | hard caps, default 1.0. URAM is inert. |
| `--force-dsp` | forced-DSP calibration factors (`finn_cost_model.md`; currently identity) instead of the auto-resType avg_bits table. |
| `--target-fps` | throughput target: every node ≤ `clock_mhz·1e6/target_fps` cycles (the physically meaningful rate constraint for a dataflow pipeline). |
| `--max-latency-ms` / `--clock-mhz` | hard cap on the sum of cycles — a sequential-execution proxy, not the pipeline's real latency (see "Formulation"). The builds run at 100 MHz. |
| `--dsr-ratio` / `--pbi-ratio` | see Rate coherence; independent flags, either/both/neither may be set. |
| `--force-serial` | PE=SIMD=1 everywhere, thresholds included. |
| `--allow-lut-mult` | enables `hls_lut_noact0`. |
| `--require-simd-ge-pe` | drops conv folds with PE>SIMD — a zero-fit fix for the real PE>SIMD LUT blowup (36% of rows, 70.7% of real LUT). Never empties a fold set (PE=1 always pairs with max SIMD). Threshold nodes are unaffected. |
| `--time-limit` / `--gap-rel` | CBC limits (default 1800 s, 2%). |
| `--pin-bits-file` | TEST-ONLY: pin `y` to a `layer_bits_*.json`, skipping the bit-choice search; folding is still solved. |

## Outputs

Next to `--out-file`:
- `layer_bits_folding_*.json` — the result:
  - `layer_weight_bits`, `layer_act_bits` — per conv/pool layer.
  - `per_layer` — chosen `pe`, `simd`, `thr_ram_style`, `variant`, bits, the
    full cost dict, plus `lut_calibrated` / `bram18k_calibrated`.
  - `extra_nodes` — per extra dataflow node: `kind`, `stage`, `pe`, `simd`,
    `ram_style`, bits (`weight_bits`/`act_bits`, `bits_rule`, `bit_sources`),
    `cycles`, calibrated LUT/BRAM, DSP, URAM.
  - `dataflow_graph` — `input` (C,H,W), `edges` (predecessor lists over every
    hardware node), `shapes`, `op_types` (FINN op names), `kinds`.
  - `_diagnostics` — totals (incl. extra nodes), `extra_by_kind` breakdown,
    % of budget, constraint counts, solver settings, `branch_imbalance`
    (see below).
- `pruning_<stem>.json` (`milp_outputs.py`) — `candidates`: blocks holding
  zero-sensitivity layers (prunable ones first); `ranked_prunable_blocks`:
  every identity-skip residual block ranked by summed |trace|, with the LUT /
  BRAM / DSP / cycles pruning it would free (incl. its thresholds, add and
  fork) and the
  `ENET_PRUNED_BLOCKS` value. Only identity-skip residual blocks are marked
  prunable — the only case `apply_block_pruning` handles soundly.
- `final_output.onnx` (`milp_outputs.py`) — the solved graph for Netron:
  one node per hardware node, `op_type` set to the real FINN v0.10.1 custom-op
  name (`MVAU_rtl`/`MVAU_hls` per the layer's own chosen variant,
  `Thresholding_rtl`, `AddStreams_hls`, `DuplicateStreams_hls`,
  `StreamingConcat_hls`, `UpsampleNearestNeighbour_hls`,
  `StreamingMaxPool_hls` — see the "Dataflow graph" table above), custom
  domain `finn_milp`, attributes `pe`, `simd`, bits, `cycles`,
  `ii_cycles_per_pixel` (cycles / output pixels, NOT channel-normalized),
  `rate_elems_per_cycle`, `chain_rate_cycles_per_elem` (cycles / (C·H·W) —
  channel-normalized, exactly `--dsr-ratio`'s own `rate_expr`, directly
  checkable node-by-node against `ratio · slowest-descendant-value`),
  `pct_of_slowest_node`, `is_slowest_node`, LUT, BRAM, DSP, URAM, and (convs)
  `mvu_cycles` / `swu_cycles` / `thr_pe`. The graph doc_string carries the run
  summary. Written unconditionally on every Optimal solve, into the same
  directory as the other per-run outputs (always named `final_output.onnx`,
  not stem-derived — one file per run directory).
- `_diagnostics.branch_imbalance` (`milp_outputs.compute_branch_imbalance_report`)
  — cumulative fork-to-join cycle-SUM ratio (`max(branch sum)/min(branch
  sum)`) for every simple 2-branch diamond (`find_fork_join_diamonds` in
  `layer_topology.py`: fork → {branch A, branch B} → join, both branches
  sharing the same nearest fan-out≥2 ancestor). **2026-09-29: this is now
  exactly what `--pbi-ratio` enforces pre-solve** (previously it was
  diagnostic-only, tracking a *different*, weaker single-node join-balance
  check — see "Rate coherence" above for why that was corrected). Still
  reported post-solve here as a real achieved-value check, and still doesn't
  capture sliding-window/buffer FILL LATENCY (e.g. ConvolutionInputGenerator
  needing to buffer several image rows before its first output) — only sums
  each node's own steady-state `cycles`, a latency proxy for a pipelined
  design, not an exact fill-latency model. A diamond can still show a ratio
  above the enforced `--pbi-ratio` if neither of its branches has any real
  folding freedom (exempt from the constraint, see "Rate coherence") —
  confirmed real: the `down2` maxpool/conv fork stayed at 1.8x under
  `--pbi-ratio 1.5` while every other (foldable) diamond tracked down to
  1.5x exactly.
- `summary.csv` + `run_args.json` — one row, written (overwritten) fresh each
  run (incl. `branch_imbalance_n_diamonds`/`_median_ratio`/`_max_ratio`); no
  sweep dimension since `alpha` was removed (see "History") — a new run in
  the same `--out-file` directory simply replaces the prior one.

Regenerate the pruning report / ONNX / `_diagnostics.branch_imbalance` for an
existing result (also writes `branch_imbalance` back into `--result` itself,
so an older artifact solved before this diagnostic existed gets it too):
`python MILP/milp_outputs.py --result <json> --sensitivity-file <json>`.

## Scope boundary

The ILP assigns one act_bits per conv layer (its input stream). LayerQuantENet
quantizes per site (reduce.2 / conv_bn_act.2 / residual_add / out_act …);
`expand_layer_bits.py` bridges the two. The output is a steering signal at this
cost model's calibration, not a certified hardware guarantee.

## History

- 2026-09-17: became self-contained (was `joint_bits_folding_ilp_perlayer.py`,
  a per-layer reindexing of the per-block `joint_bits_folding_ilp.py`);
  helpers inlined; per-block files and non-S12 configs moved to `archive/`.
  Why per-layer: a block sharing one bit pair wastes accuracy headroom on its
  least-sensitive layer and budget on its most-sensitive one.
- 2026-09-17: DSP became a hard constraint (an uncapped `alpha=0.25` solve
  needed 2,564 DSP48 on a 1,728-DSP device); URAM capped too.
- 2026-09-18: URAM retired everywhere; `thr_ram_style` block/distributed.
- 2026-09-26: residual-join threshold nodes; dataflow graph; rate coherence
  on it with the O(edges) encoding; `--max-join-imbalance-ratio` removed;
  `pruning_*.json` and `dataflow_*.onnx` outputs; comments moved here.
- 2026-09-26 (later): every remaining FINN node type added as an extra node
  (AddStreams, DuplicateStreams, StreamingConcat, UpsampleNearestNeighbour,
  pad-MVAU, InitialBlock thresholds) plus FMPadding inside conv cost, with
  folding taken from FINN v0.10.1 code. Same caps, alpha 0: min summed cycles
  16.65M → 23.6M.
- 2026-09-26 (later still): `dataflow_<stem>.onnx` renamed `final_output.onnx`
  (one fixed name per run directory); its node `op_type` changed from a
  display label (`MVAU`, `Thresholding`, ...) to the real FINN v0.10.1
  custom-op name, `EXTRA_OP_LABEL`'s values updated to match (`finn_milp.py`
  itself was already unconditional about writing it on every Optimal solve —
  no CLI flag ever gated it).
- 2026-09-28: extra-node cost loop stopped applying `calibrated_lut`/
  `calibrated_bram18k` to `STREAM_NODE_KINDS` (`add`/`dup`/`concat`/
  `upsample`) — those four go through `extra_node_options`'s fixed RTL
  variant same as every other extra node, so `_calibration_force_dsp` always
  took the force_dsp branch for them too, meaning any refit of
  `_FORCED_DSP_LUT_FACTOR`/`_FORCED_DSP_BRAM_FACTOR` (see finn_cost_model.md's
  "S12 dense 256x256 v2 refit") would have silently applied a factor fit only
  from real MVAU/VVAU/SWU/threshold data to the FINN-estimate-based
  `_ADDSTREAMS_LUT_PER_PE`/`_DUPSTREAMS_LUT_PER_PE` constants, which have no
  real Vivado basis at all yet. Those four kinds now use their raw
  (unmultiplied) cost unconditionally; everything else (per_layer entries,
  standalone join-thresholds, pad_mvau) is unaffected and still calibrated as
  before.
- 2026-09-28 (later): main per-layer cost loop's MVAU/VVAU weight-tile
  `ram_style` changed from hard-coded `RAM_STYLE_BLOCK` to `RAM_STYLE_AUTO`
  (`pad_mvau`'s extra-node cost too). The old code's comment claimed
  block-always was "what real hardware does" — real per-node Vivado data
  (`MILP/calibration.csv`'s `mvau_weight_bram` row) shows that was wrong:
  179/189 real MVAU_rtl/VVAU_hls nodes actually built with `ram_style=auto`
  land in LUTRAM (0 real BRAM), which `RAM_STYLE_BLOCK`'s always-nonzero
  prediction never captured. See `finn_cost_model.md`'s new `RAM_STYLE_AUTO`
  section for the empirical threshold (`_WM_BRAM_AUTO_MIN_WMEM`/
  `_WM_BRAM_AUTO_MIN_MEM_WIDTH`) this relies on.
- 2026-09-28 (later still): `alpha` removed. Every deployed run already used
  `alpha=1.0` (see the removed `--max-latency-ms` epsilon-constraint note
  above), which zeroed the folding variables' objective weight entirely —
  the folding a solve landed on was therefore always a solver-arbitrary tie
  among feasible options, not a real speed decision, and inspecting a
  solved `S12_dense_nn_upsample_256_w8_16_v4` (`--hard-lut-fraction 0.5
  --hard-bram-fraction 0.2 --hard-dsp-fraction 0.9 --target-fps 200`) showed
  exactly that: PE=1 on 78/90 conv/pool layers and all 81 threshold nodes,
  LUT/BRAM pinned at their 50%/20% caps (spent on bit-width) while DSP sat at
  38.8% of a 90% allowance. Objective is now `mean(sens_norm)` only; `z`
  never appears in it. CLI `--alpha` removed; `solve_joint_perlayer` no
  longer takes an `alpha` argument; result dicts (top-level and
  `_diagnostics`) no longer have an `alpha` key; `summary.csv`/
  `run_args.json` write a single row per run instead of upserting one row
  per alpha (`_update_sweep_summary` renamed `_write_run_summary`). Hard
  resource/throughput/rate-coherence/latency constraints are unchanged.
- 2026-09-29: `--optimize-downstream-rate` split back into `--dsr-ratio`
  (chain coherence, `dsr_ratio` parameter) and `--pbi-ratio` (join balance,
  `pbi_ratio` parameter) so the two mechanisms can be held fixed
  independently while sweeping the other — a real experiment design need
  the single merged flag couldn't express. Each defaults to `None`
  (off); either, both, or neither may be set; `topology` (predecessor_map/
  dataflow_map) is resolved once and shared by both gated blocks. Also
  corrected this doc's chain-coherence formula, which had L and D on the
  wrong sides of the inequality (see "Rate coherence" above) — the code's
  actual behavior did not change, only the documentation of it.
- 2026-09-29 (later): `--pbi-ratio`'s join-balance constraint corrected to
  compare cumulative fork-to-join branch SUMS instead of only the single
  node immediately before a join. Found while sweeping `--pbi-ratio` to its
  tightest possible value (1.0, exact single-node equality) on the
  `nearest_upsample` architecture and observing zero effect on
  `milp_outputs.compute_branch_imbalance_report`'s own cumulative ratio
  (stuck at 1.8x throughout) — the old constraint was structurally blind to
  a branch having more pipeline stages than its sibling (here: `skip_quant`,
  1 op, vs. `reduce`→`conv`→`expand`, 3 ops, feeding the same residual add),
  since it never looked at any node but the last one on each side. Fixed by
  promoting `compute_branch_imbalance_report`'s own diamond-detection and
  cumulative-sum logic from a diagnostic into the real constraint, via a new
  shared `find_fork_join_diamonds` (`layer_topology.py`) so the enforced
  constraint and the post-solve report can never diverge on what counts as a
  diamond again. Confirmed the fix: `--pbi-ratio 1.5` now yields a real
  diamond ratio of exactly 1.5x wherever a diamond has folding freedom on
  both branches (previously unresponsive to the ratio value at all); a
  fixed-cycle diamond (`down2`'s maxpool/conv fork) correctly stays exempt
  and reports its own natural 1.8x, unaffected.
