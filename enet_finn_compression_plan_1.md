# ENet → FINN Compression Plan (Coronary Vessel Segmentation)

Each stage is a **checkpoint**: results inspected and next stage decided before it
runs. Not a single unattended experiment.

## Decisions
1. **Binary segmentation** (vessel vs background) on ARCADE standard splits (train, val, test). Decouples segmentation from
   branch typing (LAD/LCx/RCA); comparable with prior work. Typing deferred to
   post-processing on the extracted tree if time allows.
2. **FINN** as primary toolchain: Le Blevec reports best ENet performance via
   FINN; per-layer folding is finer than hls4ml's global reuse factor;
   PyTorch-native (matches nnU-Netv2); layer-by-layer architecture allows DDR
   buffering; AMD-supported. hls4ml tried only if time permits.
3. **Target: ZU7EV.**

## Structural vs. capacity decisions (ordering principle)
- **Structural** = changes what the network *is* (decoder topology). Fixed
  **early**, because all capacity tuning is optimised against it.
  → **max-unpool vs Upsample+conv** is structural: it alters decoder information
  flow *and* Upsample+conv **adds parameters**, breaking any param budget set
  before it. Decided in **Stage 1b**.
- **Capacity** = how much network there is. Filters and bottlenecks are swept
  **jointly** (Stage 2), bit-width separately (Stage 4).
- **Within-block op swaps** (dilated / asymmetric / strided) are cheap,
  FINN-native either way, and don't change topology → deferred to Stage 3.

## Metrics
- **Dice — primary.** All goals and gates defined on Dice.
- **clDice, connected-component count — tracked every run, not gated.** ARCADE
  annotates SYNTAX-scoreable segments, so absolute connectivity values are soft;
  tracking preserves the analysis without the search depending on it.
- **params + FLOPs (MACs)** every run.

## Efficiency currency
- FP32 stages: **params + FLOPs**.
- Quantized stages: **BOPs = MACs × bits**, **params × bits**.

## Channel-count constraint
ENet's bottleneck projects to `C/4`, so counts should be **divisible by 4**;
**round up**. Reason beyond code hygiene: FINN folding needs PE | out-channels
and SIMD | in-channels, so divisor-rich counts give latency-tuning freedom in
Stage 5.
> **Floor note:** at 4 channels the `C/4` projection yields 1 internal channel
> (4→1→1→4) — the bottleneck's degenerate limit. Requires `max(1, ·)` clamping.
> Folding freedom is minimal at this width (PE=1), so U32 is a floor-finding
> point rather than a serious candidate.

---

## Stage 0 — Pipeline validation
- **Smoke test:** full pipeline, tiny subset, few epochs, end-to-end.
- **Metric sanity check:** Dice + clDice on a synthetic connected mask vs. the
  same mask with an induced gap; confirm expected behaviour.

---

## Stage 1 — Goal definition

| Run | f_i | f1 | f2 | f3 | f4 | f5 | Bnecks/stage | Unpool | Params | FLOPs | Dice | clDice | #Comp |
|-----|-----|----|----|----|----|----|--------------|--------|--------|-------|------|--------|-------|
| ENet paper-faithful | 16 | 64 |128 |128 | 64 | 16 | 5/8/8 | yes | | | | | |
| E1 (tuned baseline) | 20 | 72 |144 |144 | 72 | 20 | (yours) | yes | | | | | |

Reference = best → `D0` (Dice), `F0` (FLOPs). Record `C0` (clDice).

**Goals for final model:**
- (b) **Dice ≥ 0.9·D0**
- (c) params < 75k
- (d) FLOPs ≤ 0.3·F0
- (e) fully on-chip on ZU7EV
- (f) latency < 10 ms/frame

**Out of scope:** power optimisation; resource-util optimisation (need only
*fit*); proving per-param/FLOP optimality; generalizable sensitivity study;
modality-specific training strategy.

---

## Stage 1b — Decoder topology
Q: max-unpool or Upsample+conv? Fixed here so Stage 2 optimises against a settled
topology.

| Run | decoder | Params | FLOPs | Dice | clDice | #Comp | FINN native |
|-----|---------|--------|-------|------|--------|-------|-------------|
| E1 + unpool | max-unpool | | | | | | **no** |
| E1 + upsample | Upsample(2)+conv | | | | | | yes |

**Decision rule:** choose Upsample+conv unless max-unpool wins by a margin that
justifies (a) adding FINN support for it and (b) the parameter delta. Note
Upsample+conv *adds* params — record the delta, it shifts the Stage-2 budget.

**Checkpoint** → decoder fixed for all downstream stages.

---

## Early probes — run in parallel with Stage 2 (de-risking)

Not on the critical path. Both retire risks that would otherwise surface only at
the very end.

### P1 — FINN resource feasibility probe
**Risk retired:** the Stage-1 goals are set without deployment evidence. If wrong
by a large factor, that is only discovered in Stage 5, after every sweep has been
interpreted against them.

Take one mid-range config (e.g. U8) at INT8, push through FINN to the
**analytical resource/throughput estimation** step (no full synthesis), compare
against ZU7EV.

| Config | bits | Est. LUT | Est. BRAM | Est. DSP | Est. latency | Fits ZU7EV? |
|--------|------|----------|-----------|----------|--------------|-------------|
| U8 | 8 | | | | | |

**Outcome:** confirms or re-targets the param/FLOP budget *before* the sweeps are
interpreted. Cost ~1 day, no training.

### P2 — Early quantization probe
**Risk retired:** quantization is where the FPGA win comes from, but its behaviour
on this task is otherwise unknown until Stage 4. Since quantization is
near-orthogonal to structure, it can be characterised early.

| Config | bits | Params×bits | BOPs | Dice | clDice | #Comp |
|--------|------|-------------|------|------|--------|-------|
| E1 or U4 | 8 | | | | | |
| | 4 | | | | | |
| | 2 | | | | | |

**Outcome:** tells us early whether aggressive quantization is viable, which
informs how much parameter budget is actually needed. Scouting run — Stage 4
still runs properly on the final architecture.

---

## Stage 2 — Architecture grid (filters × bottlenecks, joint)

Filters and bottlenecks are swept **jointly as a 2D grid**, not sequentially.
All runs are independent → one parallel batch, one wall-clock wait.

**Why joint rather than sequential:** filter count and bottleneck count are both
capacity knobs acting in the same direction, so a greedy filters-then-bottlenecks
path can land off the joint optimum. The grid **eliminates** that ordering risk
rather than mitigating it, and subsumes what would otherwise be a separate
filter re-check.

**2.1 Fix f_i = 20** — isolates downstream stages from feature extraction.

**2.2 Cost tables** — architecture-only, no training. Run first; they inform the
optional fine-tuning steps and the write-up.

*Filter marginal cost:*

| Stage | +Δfilters | Δparams | ΔFLOPs | Notes |
|-------|-----------|---------|--------|-------|
| f_i | | | | high-res → FLOP-heavy? |
| f1 | | | | |
| f2 | | | | |
| f3 | | | | widest → param-heavy? |
| f4 | | | | |
| f5 | | | | high-res |

*Bottleneck marginal cost:*

| Stage | +1 bottleneck | Δparams | ΔFLOPs | Notes |
|-------|---------------|---------|--------|-------|
| s1 | | | | |
| s2 | | | | |
| s3 | | | | narrowest spatial → cheap FLOPs? |
| s4 | | | | |
| s5 | | | | high-res → FLOP-heavy? |

**2.3 The grid** — filter configs (f_i = 20 fixed) × bottlenecks per stage:

Filter axis (rounded up to mult-4):

| Name | factor | f_i | f1 | f2 | f3 | f4 | f5 |
|------|--------|-----|----|----|----|----|----|
| E1 | 1.0 | 20 | 72 |144 |144 | 72 | 20 |
| U2 | 0.5 | 20 | 36 | 72 | 72 | 36 | 12 |
| U4 | 0.25 | 20 | 20 | 36 | 36 | 20 | 8 |
| U8 | 0.125 | 20 | 12 | 20 | 20 | 12 | 4 |
| U16 | 0.0625 | 20 | 8 | 12 | 12 | 8 | 4 |
| UF | ~0.03 | 20 | 4 | 8 | 8 | 4 | 4 |

† floor-finding, bottleneck at degenerate limit.

Bottleneck axis: **{ENet-native, 5, 3, 2}**

Grid = filter configs × bottleneck settings. Record every cell:

| Run | filters | bnecks | Params | FLOPs | Dice | clDice | #Comp | Meets goals? |
|-----|---------|--------|--------|-------|------|--------|-------|--------------|
| | U2 | 5 | | | | | | |
| | U2 | 3 | | | | | | |
| | U2 | 2 | | | | | | |
| | U4 | 5 | | | | | | |
| | U4 | 3 | | | | | | |
| | U4 | 2 | | | | | | |
| | U8 | 5 | | | | | | |
| | U8 | 3 | | | | | | |
| | U8 | 2 | | | | | | |
| | U16 | 5 | | | | | | |
| | U16 | 3 | | | | | | |
| | U16 | 2 | | | | | | |
| | UF | 3 | | | | | | |
| | (E1 / native as reference) | | | | | | | |

→ config meeting goals with lowest params/FLOPs = **`Cfinal_arch`**.

**2.4 f_i reduction** *(secondary / optional)* — on the chosen grid point:

| Run | f_i | rest = Cfinal_arch | Params | FLOPs | Dice | clDice | #Comp |
|-----|-----|--------------------|--------|-------|------|--------|-------|
| base | 20 | | | | | | |
| | 12 | | | | | | |
| | 8 | | | | | | |
| | 4 | | | | | | |

Consider repeating this for two points, e.g. U16 and U14 to allow coarse interpolation. Do smaller stages become less sensitive to Fi? 

**2.5 Heterogeneous bottleneck allocation** *(secondary / optional)* — hold total
constant, redistribute across stages; tests ENet's "large encoder, small decoder"
rationale.

| Run | s1 | s2 | s3 | s4 | s5 | Total | Params | Dice | clDice |
|-----|----|----|----|----|----|-------|--------|------|--------|
| uniform (from grid) | | | | | | | | | |
| encoder-heavy | | | | | | (same) | | | |
| decoder-heavy | | | | | | (same) | | | |

**2.6 Single-stage fine-tune** *(contingency only, if grid misses goals)* — use
2.2 cost tables; adjust one stage per run; document each deviation.

**Checkpoint** → `Cfinal_arch`.

---

## Stage 2b — Seed variance *(deferred; run after Stage 2, budget permitting)*
**Why:** every grid cell is a single-seed point estimate, and ~15 configs are
compared, so the winner is partly selection luck. Without a variance estimate,
small differences between adjacent cells are uninterpretable.

**Deferred deliberately** until the grid is complete, in case it consumes more
budget than expected.

| Config | seed 1 | seed 2 | seed 3 | mean | std |
|--------|--------|--------|--------|------|-----|
| E1 | | | | | |
| Cfinal_arch | | | | | |
| (one adjacent grid cell) | | | | | |

**Use:** state the noise floor — *"seed-to-seed std is ±X Dice; differences below
X are not interpreted."* Re-read the grid against it; if the chosen cell's margin
over its neighbour is within noise, either is defensible and cost breaks the tie.

---

## Stage 3 — Within-block op swaps
Q: which non-standard conv types are worth keeping? All FINN-native (RTL SWG:
strided, dilated, non-square, depthwise) → pure performance decision.
*(Decoder topology already settled in Stage 1b — not revisited.)*

**Conditional:** only run if `Cfinal_arch` misses the FLOP or fit target and needs
the savings, or if the ablation table is wanted for the write-up.

**3.1** Estimate params / FLOP / resource cost per op.
**3.2** Ablate one op per run:

| Run | op disabled | replaced with | ΔParams | ΔFLOPs | Dice | clDice | #Comp |
|-----|-------------|---------------|---------|--------|------|--------|-------|
| baseline | — | — | | | | | |
| no-dilated | dilated | regular 3×3 | | | | | |
| no-asymmetric | 5×1, 1×5 | 3×3 | | | | | |
| no-strided | strided | maxpool+conv | | | | | |

**3.3** Keep an op only if it earns its cost given Stage-1 goals.

**Checkpoint** → `Cfinal_ops`.

---

## Stage 4 — Quantization (Brevitas QAT)
Q: how do homogeneous / heterogeneous INT8/INT4/INT2 affect Dice, and which
scheme meets the goals? (P2 already gives an early read; this is the proper run
on the final architecture.)

**4.1 Homogeneous sweep:**

| Run | bits | Params×bits | BOPs | Dice | clDice | #Comp |
|-----|------|-------------|------|------|--------|-------|
| FP32 | 32 | | | | | |
| Q8 | 8 | | | | | |
| Q4 | 4 | | | | | |
| Q2 | 2 | | | | | |

clDice tracked alongside — if the curves diverge, document it.

**4.2 Per-layer heterogeneous** — quantize one layer at a time, rest FP32:

| Run | quantized layer | bits | BOPs | Dice | clDice | #Comp |
|-----|-----------------|------|------|------|--------|-------|
| | f_i | | | | | |
| | f1 | | | | | |
| | f2 | | | | | |
| | f3 | | | | | |
| | f4 | | | | | |
| | f5 | | | | | |

**4.3** Pick Pareto-optimal (Dice vs. BOPs) meeting goals. Export QONNX.

> Escape hatch: if a different grid cell quantizes markedly better, revisit
> Stage 2 rather than freezing `Cfinal_ops`.

---

## Stage 5 — Hardware tuning (FINN folding)
Per-layer PE×SIMD for latency < 10 ms while fitting ZU7EV. Balance the pipeline
(match initiation intervals).

| Config | folding notes | LUT | FF | DSP | BRAM | Latency ms | Fits ZU7EV? |
|--------|---------------|-----|----|----|------|-----------|-------------|
| | | | | | | | |

Include the per-image z-norm frame buffer in the latency budget.

---

## Limitations

**1. Decoupling and stage ordering.** Sequential optimisation
(topology → capacity → ops → quantization) may not reach the global optimum.
Grounds for this order:

- *Structural before capacity.* Decoder topology (Stage 1b) is fixed first
  because it changes what the network is and alters the parameter budget; late
  removal would invalidate Stage 2.
- *Joint capacity sweep.* Filters and bottlenecks are swept **together** as a
  grid, so the filter↔bottleneck ordering risk is eliminated, not merely
  mitigated.
- *Near-orthogonality.* Capacity changes structure; quantization changes the
  precision of that structure. Largely independent axes.
- *Magnitude ordering.* Within-block op swaps are small corrections relative to
  capacity and bit-width, applied late against an otherwise-fixed model.
- *Search-space reduction.* Full joint search over all four axes is
  combinatorial; the remaining decoupling is standard practice (Ghielmetti et al.
  likewise fixed structure before searching quantization).

*Mitigations:* Stage 1b removes the structural-reordering risk; the Stage-2 grid
removes the filter↔bottleneck risk; Stage 4 has an explicit escape hatch.

**2. Residual structure↔quantization interaction.** The grid optimum is found in
FP32; a different cell might quantize better. Mitigated by P2 (early read) and
the Stage-4 escape hatch, not eliminated.

**3. Coarse grid.** The filter axis is geometric (×0.5 steps); the optimum may
fall between rungs. Mitigated by 2.6 (contingency fine-tune) and the 2.2 cost
tables.

**4. Arbitrary goals.** Thresholds set before deployment evidence exists.
Partially mitigated by **P1** (early FINN resource probe), which tests whether the
budget is realistic before the grid is interpreted; Stage 5 validates fully.

**5. Params ≠ compute.** Addressed by carrying FLOPs (FP32) and BOPs (quantized).

**6. clDice on scope-limited labels.** Reliable for *relative* comparison
(identical labels throughout), not as absolute connectivity statements. Hence
tracked, not gated.

**7. Single-seed comparisons.** The grid compares point estimates across ~15
configs, so the winner is partly selection luck. Addressed by **Stage 2b**
(deferred); until then, small differences between adjacent cells should not be
over-interpreted.

---

## Training protocol (all stages)
- Full **150 epochs** per run. Log Dice/clDice curves; flag any model still
  improving at 150 (under-trained → unfair comparison).
- Fixed across runs: optimizer, LR schedule, augmentation, loss, **image-level**
  split, seed. Only the variable(s) under study change.
- **Normalisation: per-image z-norm**, matching the deployment scenario (real-time
  frames from the X-ray detector). Stage 5 note: per-image statistics need the
  full frame before normalising (one-frame buffer / two-pass) — acceptable for
  frame-wise acquisition, but account for it in the latency budget.
- Channel counts divisible by 4 (round up).
