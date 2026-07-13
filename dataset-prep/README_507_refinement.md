# Dataset507_ARCADE_refinement

Training data for a **second-stage refinement network**: given ENetOriginal's
first-pass prediction plus the original input image, learn to correct it
against GT. Built by `prepare_arcade_507_refinement.py`.

(Note: this is Dataset**507**, not 506 -- `Dataset506_stn_stx_oversampled`
already exists on the `patch-enet` branch, built earlier in this same
session, so 507 was used instead to avoid clobbering it.)

## Recipe

1. **Source**: `Dataset501_ARCADE` (the original full-size 512x512, 4-class
   `{background, LAD, RCA, LCX}` dataset) plus
   `nnUNetTrainerENetOriginal`'s own checkpoint
   (`nnUNetTrainerENetOriginal__nnUNetPlans__2d/fold_0/checkpoint_best.pth`).
2. **Inference**: `nnUNetv2_predict` run on Dataset501's `imagesTr` (all
   1000 train + 200 val cases together, split by `splits_final.json`
   afterward) and `imagesTs` (300 test cases, already existed as
   `labelsPr_ENetOriginal` from an earlier run). Both are binarized (any
   class > 0 = vessel) before anything else -- this dataset only cares about
   vessel-vs-background, not per-territory class identity.
3. **Discontinuity ("residual") detection**: `fn = gt_binary & ~pred_binary`
   -- pixels GT says are vessel that the model missed. Connected-component
   labeled; components with fewer than `MIN_DISCONTINUITY_PX` (15) pixels
   are dropped as noise, not real gaps. Up to `MAX_DISCONT_PER_IMAGE` (8)
   per source image, largest first.
4. **Patch extraction**, three categories per source image, all 85x85:
   - **discontinuity** -- one patch centered on each kept FN connected
     component (from step 3). These are the model's *actual observed*
     failures, as opposed to `VesselGapTransform`'s synthetic contrast-drop
     gaps -- real low-contrast/interpolation failures, not simulated ones.
   - **normal** -- skeletonize + farthest-point-sample coverage of the GT
     mask (same technique as Dataset505/506), **filtered to points where the
     prediction already matches GT well** (FN fraction inside the patch
     &le; 10%). Without this filter, "normal" ends up sampling failures just
     as often as successes on a badly-mispredicted image and stops being a
     meaningful contrast against the discontinuity category -- caught this
     empirically while validating the pipeline (see git history).
   - **empty_or_fp** -- random locations that are either GT-empty or
     FP-dominated (&ge;5% of the patch is predicted vessel where GT has
     none). Teaches "don't hallucinate vessel," the complementary failure
     mode to discontinuities.
5. **Balancing**: parameterizable percentages (default 25% empty_or_fp / 50%
   normal / 25% discontinuity, must sum to 1.0). The discontinuity pool is
   the anchor -- it's the scarcest category and the entire reason this
   dataset exists -- so total patch budget is derived from
   `n_discontinuity / discont_frac`, then `normal`/`empty_or_fp` are randomly
   subsampled down to their target share of that total (files for anything
   not selected are deleted after the fact, since every candidate patch gets
   written to disk during extraction).
6. **Train+val vs test**: source `train_*`/`val_*` images produce
   Dataset507's `imagesTr`/`labelsTr` (own train/val split preserved from
   Dataset501's `splits_final.json`, applied per source image); source
   `test_*` images produce a separate `imagesTs`/`labelsTs`, balanced
   independently so its composition doesn't skew (or get skewed by)
   `imagesTr`. Test patches are never mixed into the training pool -- same
   held-out-set discipline as every other dataset in this repo.

## Format

Each case has **two input channels**, following nnU-Net's multi-channel
convention, plus a binary GT label:

| file | content |
|---|---|
| `{case_id}_0000.png` | raw grayscale input patch |
| `{case_id}_0001.png` | ENetOriginal's predicted mask patch (0/255) |
| `{case_id}.png` (labelsTr/labelsTs) | GT vessel mask patch (0/1) |

`dataset.json` declares `channel_names: {"0": "grayscale", "1":
"predicted_mask"}`.

## Case ID naming

`{source_stem}_{category}_{idx:03d}`, e.g. `train_183_discont_002`,
`val_42_normal_000`, `test_277_empty_001`. `source_stem` already encodes
which Dataset501 split the patch came from (`train_N`/`val_N`/`test_N` is
Dataset501's own file naming, not something added on top) -- so every case
ID is traceable to both its source image and why it was selected, without
needing to consult any side table.

## Usage

```bash
python prepare_arcade_507_refinement.py \
  --dataset-id 507 --patch-size 85 \
  --empty-frac 0.25 --normal-frac 0.50 --discont-frac 0.25 --seed 0
```

Requires `Dataset501_ARCADE/labelsPr_ENetOriginal_Tr` (train+val inference)
and `labelsPr_ENetOriginal` (test inference) to already exist -- run
`nnUNetv2_predict` first if not (see `run_enetoriginal_inference.sh` or just
point it at `imagesTr`/`imagesTs` with `-tr nnUNetTrainerENetOriginal`).

See `preview_dataset507.ipynb` for example patches from each category and
summary statistics.
