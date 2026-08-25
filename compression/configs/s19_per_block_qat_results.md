# S19 — Per-Block HAWQ QAT Results

Quantization-aware training of `nnUNetTrainerENet_19_reginterleaved_separable_nonneg_block_double_mid`
("S19") at a HAWQ-searched **per-bottleneck-block** (not per-stage) mixed
weight/activation bit-width, using a genuinely trained `nonneg_block`
decomposed-PReLU activation (not a post-hoc approximation).

## At a glance

| | FP32 source (S19) | Per-block QAT (this result) |
|---|---:|---:|
| Dice (mean, 4-class, official protocol) | 0.7931 | **0.7458** |
| Params (real, deployed model) | 30,948 | 31,324 |
| MACs (see note 1 below on methodology) | 272,564,224 | 171,704,320* |
| Weight/activation bit-width | 32/32 (FP32) | mixed 2–8 / mixed 2–8 (per block) |
| Effective average bit-width (BOPs / MACs) | 32.0 | 16.55 |

\* MACs are architecture-determined (topology-only, independent of
quantization) — the FP32 and quantized rows differ here only because they're
computed by two different tools; see Note 1.

## 1. Architecture (FP32 source)

Byte-identical to `compression/hawq/config_23_1.py`'s own architecture (23_1
is a QAT continuation warm-started from this exact S19 checkpoint):

| | |
|---|---|
| Channels (initial, stage1, stage2/3, stage4, stage5) | 4, 16, 32, 16, 4 |
| Bottlenecks per stage (stage1, stage2, stage3, regular4, regular5) | 4, 12, 12, 2, 1 |
| Decoder type | `upsample_conv` |
| Context pattern | `dense_dilation_reg_interleaved_double_mid` (S19's own pattern: reg,2,4,8,16,reg,reg,2,4,8,16,reg — 12 slots, the interior reg-bookend doubled) |
| Asymmetric convs | off |
| Separable-dilated | on (every dilated 3×3 factored into a (3,1)+(1,3) pair) |
| DSC / dsc_no_projection | off |
| PReLU variant | `nonneg_block` — one real **trained** non-negative learnable scalar shared across every activation site within a block (not per-channel) |
| Total bottleneck blocks | 37 (see `compression/hawq/block_utils.py`'s `enumerate_blocks`) |

37 blocks: `initial`, `down1`, `regular1.0-3`, `down2`, `stage2.0-11`,
`stage3.0-11`, `up4`, `regular4.0-1`, `up5`, `regular5.0`, `final`.
(`proj2_to_3` is absent — stage2/stage3 channel widths match, so ENet builds
it as `nn.Identity()`.)

## 2. Quantization scheme

**Method**: `compression/hawq/block_sensitivity.py` (Hutchinson-trace Hessian
estimation, weights + activations, 14/16 real batches used) →
`compression/hawq/finn_block_costs.py` (closed-form FINN-R LUT/BRAM cost per
block per candidate bit) → `compression/hawq/ilp_search.py` (two independent
MIPs — weight axis and activation axis — minimizing normalized sensitivity +
BRAM penalty subject to an interpolated LUT budget).

**Search settings**: candidate bits `{2, 4, 8}`; `--weight-budget-fraction 0.5
--act-budget-fraction 0.5 --bram-weight 1.0`.

**Result** (`compression/hawq/block_bits_s19.json`) — one independent
(weight, activation) bit pair per block:

| block | W | A | | block | W | A |
|---|---:|---:|---|---|---:|---:|
| initial | 4 | 8 | | stage3.0 | 2 | 4 |
| down1 | 4 | 8 | | stage3.1 | 4 | 4 |
| regular1.0 | 4 | 8 | | stage3.2 | 4 | 4 |
| regular1.1 | 4 | 4 | | stage3.3 | 4 | 4 |
| regular1.2 | 4 | 4 | | stage3.4 | 4 | 4 |
| regular1.3 | 4 | 4 | | stage3.5 | 2 | 4 |
| down2 | 2 | 4 | | stage3.6 | 2 | 4 |
| stage2.0 | 4 | 4 | | stage3.7 | 4 | 4 |
| stage2.1 | 4 | 4 | | stage3.8 | 4 | 4 |
| stage2.2 | 2 | 4 | | stage3.9 | 4 | 4 |
| stage2.3 | 2 | 4 | | stage3.10 | 4 | 4 |
| stage2.4 | 4 | 4 | | stage3.11 | 2 | 4 |
| stage2.5 | 4 | 4 | | up4 | 2 | 2 |
| stage2.6 | 2 | 4 | | regular4.0 | 4 | 4 |
| stage2.7 | 4 | 4 | | regular4.1 | 2 | 4 |
| stage2.8 | 4 | 4 | | up5 | 4 | 4 |
| stage2.9 | 4 | 4 | | regular5.0 | **8** | **8** |
| stage2.10 | 4 | 4 | | final | 4 | 8 |
| stage2.11 | 2 | 4 | | | | |

`regular5.0` is the only block kept at full 8/8 — its Hessian-trace
sensitivity (`trace_w = 2.47`) was by far the highest of all 37 blocks
(next-highest: `initial` at 0.55) despite having only 17 weight params.
`up4` is the most aggressively quantized block (2-bit weights **and**
2-bit activations).

**Activation function**: `QuantDecomposedLeakyAct` (Brevitas has no native
quantized PReLU) using S19's own **real trained** per-block `nonneg_block`
scalars — extracted losslessly via `extract_leaky_slope_map.py`
(`compression/post-quantization/slope_maps/
19_reginterleaved_separable_nonneg_block_double_mid.json`), not a post-hoc
approximation. Decoder blocks (`up4`, `regular4`, `up5`, `regular5`,
`final`) are always plain `QuantReLU` regardless (same rule the FP32
architecture itself uses).

## 3. Training (QAT)

| | |
|---|---|
| Trainer | `nnUNetTrainerENetQuantS19Block` |
| Model class | `QuantENetS19Block` (`enet/nnunetv2/nets/QuantENetS19Block.py`) |
| Warm start | S19's own FP32 `checkpoint_final.pth` (strict=False name+shape transfer — 873/1119 keys, 0 shape mismatches) |
| Epochs | 150 (target and reached) |
| Iterations/epoch | 250 (nnU-Net default, not reduced) |
| Batch size | 12 |
| Optimizer | AdamW (betas 0.9/0.999, eps 1e-8) |
| Initial LR | 1e-3, PolyLRScheduler |
| Weight decay | 1e-2 |
| Run | Two Slurm jobs: `qat_s19_block_bits.job` (cold-submitted, walltime-limited at ~10h, reached checkpoint_best epoch 135 / stale checkpoint_latest epoch 100) → `resume_qat_s19_block_bits.job` (`--c` resume, correctly promoted the further-along `checkpoint_best.pth` over the stale `checkpoint_latest.pth` first) → completed to `checkpoint_final.pth`, epoch 150 |
| Checkpoint evaluated | `checkpoint_best.pth`, epoch 147 |
| `converged_flag` | `False` (EMA dice still net-rising over the trailing window, though the last 10 logged epochs were largely flat: 0.7111 → 0.7165 → 0.7155) |

## 4. Dice (real, official protocol via `compression/collect_results.py`)

| class | Dice | clDice |
|---|---:|---:|
| LAD | 0.6328 | 0.6447 |
| RCA | 0.8372 | 0.8428 |
| LCX | 0.7590 | 0.7645 |
| LM | 0.7540 | 0.7616 |
| **Mean (4-class)** | **0.7458** | 0.7534 |
| Binary (vessel vs. background) | 0.7039 | — |
| Fragmentation (`n_components`, pooled) | 8.43 | — |

## 5. Params / MACs / BOPs

| | value |
|---|---:|
| Params (real deployed `QuantENetS19Block`) | 31,324 |
| Params (FP32 reference, same architecture) | 30,948 |
| MACs (architecture-only, quantization-independent) | 171,704,320 |
| BOPs (Σ MAC × W_bits × A_bits, per-block) | 2,841,640,960 |
| Effective average bit-width (BOPs / MACs) | 16.55 |

**Note 1 — MAC counting methodology.** This report's MAC figure
(171,704,320) comes from the same layer-geometry tracer used throughout
this project's FINN/HAWQ cost pipeline (`compression/hawq/
finn_block_costs.py`'s `dump_block_layer_geometry`, which correctly
resolves to exactly 125 real `Conv2d`/`ConvTranspose2d` modules via
PyTorch's own `named_modules()` de-duplication). This does **not** match
the MAC/FLOPs figure that `thop` reports for the same architecture
(`compression/collect_results.py`'s convention, used for every other row in
`compression/results.csv`, incl. this project's own official
`nnUNetTrainerENet_19_..._double_mid` FP32 row): thop reports 394,330,112
MACs, roughly 1.45× higher. Traced directly: `ENet.py`'s `RegularBottleneck`
builds `self.conv_bn_act = nn.Sequential(self.conv, BN, act)`, literally
reusing the `self.conv` object rather than a copy — thop's own internal
module traversal (unlike `named_modules()`'s default) does not de-duplicate
this shared submodule, so it accumulates that layer's MACs twice. This
looks like it affects every ENet/QuantENet config profiled via thop in this
repo's history, not just S19 — flagged here for visibility, not fixed
project-wide as part of this report.

## 6. FINN analytical hardware cost (post-folding)

Via `compression/hawq/folding_ilp.py --granularity block` (per-layer PE/SIMD
+ BRAM-vs-URAM `ram_style` search, LUT/BRAM as a soft objective penalty, not
a hard constraint — see that script's own docstring):

| resource | total | % of XCZU7EV budget |
|---|---:|---:|
| LUT | 102,833 | 44.6% |
| BRAM_18K | 714 | 114.4% |
| URAM (288Kb blocks) | 125 | *(96 available on XCZU7EV — see `XCZU7EV.csv`; over budget)* |
| Cycles (Σ, ≈ per-image latency) | 29,118,464 | — |

125 of 128 traced layers were assigned `ram_style=ultra` (URAM) by the
folding ILP, only 3 stayed on BRAM. DSP is **not modeled** by this repo's
closed-form cost formulae (`finn_cost_model.py`'s own documented scope
limitation) — no DSP utilization figure is available for this
configuration without an actual FINN/Vivado build.

**Caveat** (same one this repo's cost model always carries): this is a
closed-form analytical estimate, not post-synthesis Vivado numbers. Real
synthesis typically comes in under this estimate for LUT/BRAM — the 114.4%
BRAM figure and the URAM overage above should be read as directional, not
as a hard fail.

## Artifacts

- `compression/hawq/block_bits_s19.json` — per-block bit assignment
- `compression/hawq/block_sensitivity_s19.json` — raw Hessian-trace sensitivity per block
- `compression/hawq/finn_block_costs_23_1.json` — per-block FINN cost table (architecture-level, shared with 23_1)
- `compression/hawq/folding_block_s19.json` — per-layer folding/ram_style solution
- `compression/post-quantization/slope_maps/19_reginterleaved_separable_nonneg_block_double_mid.json` — real trained per-block PReLU slopes
- `compression/slurm/qat_s19_block_bits.job`, `compression/slurm/resume_qat_s19_block_bits.job` — training jobs
- `enet/nnunetv2/nets/QuantENetS19Block.py`, `enet/nnunetv2/training/nnUNetTrainer/nnUNetTrainerENetQuantS19Block.py` — model/trainer
- `compression/results.csv`, `config_name=nnUNetTrainerENetQuantS19Block_s19_qat_block_bits`, `stage=experiment_s19_qat_block_bits` — official Dice row
