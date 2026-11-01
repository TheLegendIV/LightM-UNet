# Max-unpool skip ablations (PReLU + max_unpool ENet, width sweep)

Question: why does max-unpool fail so much harder than bilinear upsampling at low channel widths?
Method: change only the skip branch of the two decoder `UpsamplingBottleneck`s (`up4`, `up5`) of the
already-trained nets at inference time, no retraining, and measure validation Dice plus skip diagnostics.

Nets: `enet_original_prelu_maxunpool_4c` (divisor 1) and `1_naive_baseline_prelu_maxunpool_{U2,U4,U8,U16}`,
`checkpoint_best.pth`, fold 0. Validation = the 200 `val` cases of `splits_final.json`.
Scoring = same per-case, per-class Dice as `nas/collect_results.py` (which scores the *test* split).

## Experiments (`exp` column)

| exp | skip branch at inference |
|---|---|
| `control` | real encoder indices (trained net, unchanged) |
| `random_shared` | one fixed random slot per 2x2 window (seed 0), the same map for every channel |
| `nearest_quarter` | no indices: v/4 written into all 4 sub-pixels (same SUM per window, no placement, full coverage; L2 norm halved) |
| `nearest_half` | no indices: v/2 written into all 4 sub-pixels (same L2 norm per window as the real skip, 4*(v/2)^2 = v^2; sum doubled) |

## Validation Dice (mean over 4 classes; one random-index draw)

| divisor | control | random indices | v/4 | v/2 (L2-matched) | control binary Dice |
|---|---|---|---|---|---|
| 1 | 0.846 | 0.838 | 0.847 | 0.845 | 0.811 |
| 2 | 0.819 | 0.814 | 0.827 | 0.838 | 0.799 |
| 4 | 0.609 | 0.597 | 0.629 | 0.616 | 0.765 |
| 8 | 0.542 | 0.535 | 0.485 | 0.534 | 0.714 |
| 16 | 0.366 | 0.353 | 0.387 | 0.339 | 0.535 |

Per-class numbers, pixel fill, norm ratios and activation stats: `dice_by_width.csv`, `stage_stats.csv`.

## What the numbers say

1. **The trained nets are insensitive to where the skip values land.** Random indices (all-or-nothing, 25%
   pixel fill) cost 0.5-1.3 Dice points at every width. The full-width net loses 0.8 points.
2. **Full coverage does not consistently help.** v/4 changes Dice by +0.1, +0.8, +2.0, -5.7, +2.1 points
   at divisors 1, 2, 4, 8, 16. The U8 drop is a LAD collapse (LAD Dice 0.274 -> 0.036), not a smooth effect.
   The L2-matched v/2 arm removes the U8 drop (-0.8 points, LAD 0.256), so that drop came from the halved
   skip norm, not from losing placement. v/2 vs control is +0.1/ +1.9/ +0.7/ -0.8/ -2.7 points: still within a
   few points, and it falls at U16 where v/4 rose, so the low-width differences are noise-level for one run.
3. **The coverage premise is real.** Measured pixel fill for `control` follows the independent-slot prediction
   1-(3/4)^C closely (stage 5, C=4: 0.62-0.66 measured vs 0.684 predicted; stage 4, U16: 0.712 vs 0.684):
   about a third of high-res pixels have no non-zero skip channel at C=4. The nets just don't visibly depend on it.
4. **The skip is not small at low widths.** log10(||main||/||out||) stays between -0.14 and -0.45 at every
   width (stage 5, U16: -0.17). For `control` and `random_shared` it is identical by construction: placement
   does not change a tensor's norm. v/4 shifts it by exactly -0.30 (norm halves).
5. **Activation statistics (1.3) carry little signal.** Curves for the three experiments nearly coincide; the
   one visible effect is v/4 roughly halving peak activations at stage 5. Minimum is 0 everywhere (see caveat 3).

## Caveats - read before using these numbers

1. **Low-width Dice is dominated by class collapse.** LCX is absent from 65% of validation cases and LM from
   35%. At U8 the nets never predict LCX or LM, and at U16 never LCX, so those classes score exactly the
   empty-case fraction (`topo.dice_score` returns 1.0 when both masks are empty): LCX = 0.650 and (U8) LM = 0.350 under all three
   experiments. The mean-Dice drop from U2 to U8/U16 is mostly classes disappearing, which these ablations do not touch.
2. **Inference-time swaps show what a trained net relies on, not what is learnable.** Any difference between
   max-unpool and bilinear that arises during training (e.g. sparse gradients into `main_proj`) is invisible here.
   The zero-skip ablation (skip removed entirely) is in `skip_ablation/` (test split, below).
3. **The decoder activations are ReLU, not PReLU.** `ENET_DECODER_PRELU` is unset in the sweep jobs, so
   `decoder_relu = True`; "PReLU" applies to the encoder only.
4. One random-index draw (seed 0); `--seeds` and `--random-mode per_channel` are supported but not run.
5. The predictor runs under CUDA autocast (fp16); stats are cast to fp32 before aggregation.

## Five-arm skip ablation on the TEST split (`skip_ablation/`)

Same nets, inference-time only, scored like `nas/collect_results.py` (300 test cases, mean per-class Dice).
Arms: `MaxUnpool` (control); `No Skip` (`main_proj(x)` replaced by zeros); `Random` (independent random 2x2 slot
per channel and window, so the per-channel sparsity/coverage of the real indices is kept, mean of 3 draws);
`v/4` (nearest, same per-window SUM); `v/2` (nearest, same per-window L2 norm).

| width | MaxUnpool | No Skip | Random | v/4 | v/2 |
|---|---|---|---|---|---|
| 1 | 0.809 | 0.785 | 0.805 | 0.806 | 0.810 |
| 1/2 | 0.788 | 0.765 | 0.781 | 0.791 | 0.798 |
| 1/4 | 0.591 | 0.607 | 0.582 | 0.607 | 0.589 |
| 1/8 | 0.502 | 0.498 | 0.498 | 0.457 | 0.495 |
| 1/16 | 0.331 | 0.329 | 0.318 | 0.352 | 0.315 |

Versus control, in Dice points: No Skip -2.4, -2.3, +1.5, -0.4, -0.2; Random -0.5, -0.7, -0.9, -0.4, -1.4
(min-max over the 3 draws is at most 0.9 pt); v/4 -0.3, +0.3, +1.6, -4.5, +2.1; v/2 +0.1, +1.0, -0.2, -0.7, -1.7.

- Removing the skip entirely costs at most 2.4 pts at any width, and nothing at 1/4 and below: the trained
  max-unpool decoders do not use their skip branch. (Bilinear nets lose 29-66% of Dice under the same test,
  see `dice_drop_vs_width.png`.)
- Random indices are slightly worse than the real ones at all five widths (-0.4 to -1.4), so real indices carry
  at most a small signal; at 1/16 random is below No Skip.
- Dense replacements (v/4, v/2) do not consistently beat the sparse control, so these tests do not show sparsity
  to be harmful. The one large deviation is v/4 at 1/8 (-4.5); not investigated.
- One trained net per width (seed 0), inference-time swaps only: this shows what these nets rely on, not what is
  learnable. Figures: `skip_ablation/dice_vs_width.png`, `dice_delta_vs_width.png`; data
  `skip_ablation/skip_ablation_results.csv`, `maxunpool_arms.csv`.

## Harness check against `nas/results.csv`

Control on the *test* split vs the stored test Dice: U1 0.8091 / 0.8091, U2 0.7878 / 0.7878, U8 0.5018 / 0.5018,
U16 0.3313 / 0.3313 (exact); **U4 0.5913 vs 0.5905** (0.0008 off, cause not investigated).
The patched forward is also checked against the original `UpsamplingBottleneck.forward` on a random input
before each width (max |diff| 0).

## Files

- `unpool_ablation.py` - runner (resume-safe; `--split`, `--divisors`, `--exps`, `--seeds`, `--random-mode`, `--limit`, `--force`).
  Run in the dev container: `docker exec lightmunet_dev python3 /workspace/LightM-UNet/report/model_search/maxUnpool/unpool_ablation.py`
- `plot_unpool_ablation.py` - plots from the CSVs
- `dice_by_width.csv`, `stage_stats.csv` - results (`split` column: `val` for the ablations, `test` for the control check)
- `exp_dice.png`, `exp_log_main_over_out.png`, `exp_pixel_fill.png`, `exp_activation_stats.png`
- `run.log` - raw runner output
