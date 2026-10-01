# Max-unpool skip ablations (PReLU + max_unpool ENet, width sweep)

Question: why does max-unpool fail so much harder than bilinear upsampling at low channel widths?
Method: change only the skip branch of the two decoder `UpsamplingBottleneck`s (`up4`, `up5`) of the
already-trained nets at inference time, no retraining, and measure validation Dice plus skip diagnostics.

Nets: `enet_original_prelu_maxunpool_4c` (divisor 1) and `1_naive_baseline_prelu_maxunpool_{U2,U4,U8,U16}`,
`checkpoint_best.pth`, fold 0. Validation = the 200 `val` cases of `splits_final.json`.
Scoring = same per-case, per-class Dice as `compression/collect_results.py` (which scores the *test* split).

## Experiments (`exp` column)

| exp | skip branch at inference |
|---|---|
| `control` | real encoder indices (trained net, unchanged) |
| `random_shared` | one fixed random slot per 2x2 window (seed 0), the same map for every channel |
| `nearest_quarter` | no indices: v/4 written into all 4 sub-pixels (same mass per window, no placement, full coverage) |

## Validation Dice (mean over 4 classes; one random-index draw)

| divisor | control | random indices | v/4 | control binary Dice |
|---|---|---|---|---|
| 1 | 0.846 | 0.838 | 0.847 | 0.811 |
| 2 | 0.819 | 0.814 | 0.827 | 0.799 |
| 4 | 0.609 | 0.597 | 0.629 | 0.765 |
| 8 | 0.542 | 0.535 | 0.485 | 0.714 |
| 16 | 0.366 | 0.353 | 0.387 | 0.535 |

Per-class numbers, pixel fill, norm ratios and activation stats: `dice_by_width.csv`, `stage_stats.csv`.

## What the numbers say

1. **The trained nets are insensitive to where the skip values land.** Random indices (all-or-nothing, 25%
   pixel fill) cost 0.5-1.3 Dice points at every width. The full-width net loses 0.8 points.
2. **Full coverage does not consistently help.** v/4 changes Dice by +0.1, +0.8, +2.0, -5.7, +2.1 points
   at divisors 1, 2, 4, 8, 16. The U8 drop is a LAD collapse (LAD Dice 0.274 -> 0.036), not a smooth effect.
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
   A zero-skip ablation (skip removed entirely) was not run; it is the most direct reliance test.
3. **The decoder activations are ReLU, not PReLU.** `ENET_DECODER_PRELU` is unset in the sweep jobs, so
   `decoder_relu = True`; "PReLU" applies to the encoder only.
4. One random-index draw (seed 0); `--seeds` and `--random-mode per_channel` are supported but not run.
5. The predictor runs under CUDA autocast (fp16); stats are cast to fp32 before aggregation.

## Harness check against `compression/results.csv`

Control on the *test* split vs the stored test Dice: U1 0.8091 / 0.8091, U2 0.7878 / 0.7878, U8 0.5018 / 0.5018,
U16 0.3313 / 0.3313 (exact); **U4 0.5913 vs 0.5905** (0.0008 off, cause not investigated).
The patched forward is also checked against the original `UpsamplingBottleneck.forward` on a random input
before each width (max |diff| 0).

## Files

- `unpool_ablation.py` - runner (resume-safe; `--split`, `--divisors`, `--exps`, `--seeds`, `--random-mode`, `--limit`, `--force`).
  Run in the dev container: `docker exec lightmunet_dev python3 /workspace/LightM-UNet/analysis/report/model_search/maxUnpool/unpool_ablation.py`
- `plot_unpool_ablation.py` - plots from the CSVs
- `dice_by_width.csv`, `stage_stats.csv` - results (`split` column: `val` for the ablations, `test` for the control check)
- `exp_dice.png`, `exp_log_main_over_out.png`, `exp_pixel_fill.png`, `exp_activation_stats.png`
- `run.log` - raw runner output
