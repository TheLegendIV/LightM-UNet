# analysis/501_ARCADE/

Evaluation tooling for ARCADE (`Dataset501_ARCADE`) predictions: standard overlap
metrics (dice/precision/recall/boundary-F1/clDice) plus diagnostics that dice
doesn't capture -- class purity, fragmentation, and anatomical-consistency
checks specific to coronary anatomy (RCA vs. LAD/LCX territory, LAD/LCX
bifurcation logic).

Classes throughout: `0=Background, 1=LAD, 2=RCA, 3=LCX`.

## Folder structure

```
analysis/501_ARCADE/
  preview_results.ipynb        Interactive notebook: inference, visual previews,
                                confusion overlays, dice/precision/recall/boundary-F1/
                                clDice, and the topology diagnostics -- all for one
                                model at a time (set NET_NAME near the top).
  segmentation_topology.py     Shared library: image/mask loading conventions used by
                                every script here, plus the territory/purity/
                                fragmentation/branch-consistency diagnostics.
  compute_topology_metrics.py  CLI: runs segmentation_topology.py over one, several,
                                or (default) all labelsPr_* folders and writes
                                per-case + per-model CSVs to results/.
  shape_diagnostics.py         CLI: a second, skeleton-branch-based purity/cleanliness/
                                catheter-FP diagnostic (see "Two purity metrics" below).
  aggregate_summary.py         CLI: rolls every {model}_overall_metrics.csv in results/
                                into results/summary_overall_metrics.csv, adding
                                mean_class_dice.
  results/                     All generated output (CSVs + confusion-grid PNGs).
                                Nothing here is hand-written -- delete and regenerate
                                at will.
  predictions/                 Reserved, currently unused.
```

Predictions themselves live outside this folder, at
`data/nnUNet_raw/Dataset501_ARCADE/labelsPr_<model_name>/`, one folder per model.
Ground truth is `data/nnUNet_raw/Dataset501_ARCADE/labelsTs/`. `<model_name>` is
what all the `{model}_*.csv` output files in `results/` are keyed on.

## Running things

```bash
# One model, interactively, with visual previews: open preview_results.ipynb,
# set NET_NAME, run all cells.

# All models, headless:
python analysis/501_ARCADE/compute_topology_metrics.py
python analysis/501_ARCADE/aggregate_summary.py          # after the notebook or the above has produced *_overall_metrics.csv
python analysis/501_ARCADE/shape_diagnostics.py --net-name ENetGlobalCtxG3
```

`compute_topology_metrics.py` and `shape_diagnostics.py` only need
`labelsPr_<model_name>/` to exist (no notebook run required first).
`aggregate_summary.py` needs `{model}_overall_metrics.csv` + `{model}_per_class_metrics.csv`
in `results/`, which only the notebook currently produces.

## Output files (per model, in `results/`)

| File | Produced by | Contents |
|---|---|---|
| `{model}_overall_metrics.csv` | notebook | one row: pooled binary (foreground-vs-background) dice/precision/recall/boundary-F1/clDice across all cases |
| `{model}_per_class_metrics.csv` | notebook | one row per class: dice/precision/recall/clDice, pooled across all cases |
| `{model}_per_image_metrics.csv` | notebook | one row per case: binary dice/precision/recall/boundary-F1/clDice |
| `{model}_confusion_grid_display.png` | notebook | visual grid (raw / GT / prediction / TP-FP-FN overlay) for a handful of sample cases |
| `{model}_topology_metrics.csv` | compute_topology_metrics.py (or notebook) | one row per case: territory leakage, component purity, fragmentation, branch consistency (see below) |
| `{model}_topology_summary.csv` | compute_topology_metrics.py (or notebook) | one row: the above averaged/counted across all cases |
| `{model}_shape_diagnostics.csv` | shape_diagnostics.py | one row per case: skeleton-branch purity, per-class fragmentation, catheter-like FP shapes (see below) |
| `{model}_shape_diagnostics_overall.csv` | shape_diagnostics.py | one row: the above averaged across all cases |
| `summary_overall_metrics.csv` | aggregate_summary.py | one row per model: everything in `*_overall_metrics.csv` plus `mean_class_dice`, sorted by `dice_f1` |
| `summary_topology_metrics.csv` | compute_topology_metrics.py | one row per model: everything in `*_topology_summary.csv`, sorted by `area_weighted_purity` |

## Metric reference

### Standard overlap (notebook, `evaluate_all()`)

- **dice_f1 / precision / recall_sensitivity** -- pooled TP/FP/FN over **binary**
  foreground (`gt>0` vs `pred>0`) across every pixel in every case.
  `dice = 2*TP/(2*TP+FP+FN)`. Because this collapses LAD/RCA/LCX into one
  "vessel" class, **it does not penalize a vessel found at the right place but
  labeled the wrong class** -- that pixel counts as a plain TP.
- **per-class dice/precision/recall** (in `per_class_metrics.csv`) -- same
  formula, recomputed per class (`gt==c` vs `pred==c`). A wrong-class pixel now
  shows up as an FN for the true class and an FP for the predicted class, so
  this *does* penalize classification error, just split across two rows.
- **mean_class_dice** (`aggregate_summary.py`) -- macro-average of the three
  foreground per-class dice scores (LAD, RCA, LCX; background excluded). Use
  this instead of `dice_f1` when comparing models on "did it get the class
  right," not just "did it find a vessel."
- **boundary_f1** -- dice computed on the 1px mask boundary (after 3x3 erosion)
  instead of the full mask, symmetric-dilated by `tolerance_px` (default 2) to
  allow near-miss boundaries. Measures contour accuracy independent of interior
  fill.
- **clDice** -- skeleton-based: precision = fraction of the *predicted*
  skeleton lying inside the GT mask; sensitivity = fraction of the *GT*
  skeleton lying inside the predicted mask; clDice is their harmonic mean.
  Rewards topological correctness (a thin vessel found end-to-end) over raw
  area overlap.

### Territory / purity / fragmentation / branch consistency (`segmentation_topology.py`)

These exist because dice treats "found the vessel, wrong class" and "found the
vessel, right class" identically, and says nothing about whether a prediction
is a single clean shape or a scatter of fragments.

- **Territory leakage** -- RCA (right coronary tree) and LAD/LCX (left
  coronary tree) never co-occur in one ARCADE image (verified: 0/300 test GT
  images are mixed). `territory_leakage_rate` = predicted-foreground pixels in
  the *wrong* territory for that image's GT, divided by total predicted
  foreground. `territory` / `cases_with_leakage` record which territory the
  image belongs to and whether any leakage occurred at all.
- **Component purity** -- label connected components of the full predicted
  foreground mask (`scipy.ndimage.label`, 8-connectivity). For each component,
  `purity` = majority-class pixel fraction. `area_weighted_purity` averages
  this across all components in all cases, weighted by component area.
  `cases_with_mixed_component` counts images with at least one blob containing
  more than one class -- this is the direct "how often does the model blend
  classes inside one connected shape" number.
- **Fragmentation** -- `pred_components` / `gt_components` = connected-component
  counts of the merged foreground mask; `component_ratio` is their ratio
  (GT is almost always 1 connected vessel tree). `pred_noise_islands` counts
  components under 5px, i.e. speckle rather than real fragmentation.
- **Branch consistency** -- skeletonizes the predicted LAD+LCX mask
  (`skimage.morphology.skeletonize`), finds tree endpoints (skeleton pixels
  with exactly one skeleton neighbor), and for every pair of endpoints in the
  same skeleton component, BFS-walks the connecting path and reads off the
  predicted class at each pixel. Runs shorter than `min_run_px` (default 3) are
  merged into a neighboring run first, to strip single-pixel skeletonization
  noise. If a class appears, is superseded by the other class, and then
  reappears later on the same path (`LAD -> LCX -> LAD`), that path is
  "inconsistent" -- physically, a vessel can only change identity once, at a
  real bifurcation, and should never revert. `cases_with_branch_inconsistency`
  counts images with >=1 such path; `branch_inconsistent_path_rate` is finer
  grained (bad paths / total paths checked) but only defined for images that
  had >=2 skeleton endpoints to trace (`n_cases_with_branch_paths`), so it's
  noisier on a small sample.
  RCA is intentionally excluded from this check: it originates from a separate
  ostium and doesn't bifurcate into LAD/LCX, so "must not revert" doesn't apply
  between RCA and the left tree the way it does between LAD and LCX. RCA/left
  mixing is instead caught by territory leakage and component purity above.

### Skeleton-branch purity / cleanliness / catheter shapes (`shape_diagnostics.py`)

A second, independent way of measuring purity and cleanliness, at finer
granularity than `segmentation_topology.py`'s blob-level purity:

- **Branch purity** -- skeletonizes the full predicted foreground, splits the
  skeleton at junctions (degree >= 3, i.e. real bifurcations) into individual
  branches, and computes majority-class purity *per branch* rather than per
  raw blob. This is more precise than component purity for the "does one
  physical vessel segment carry two labels" question, because it only measures
  mixing *within* an unbranched run -- mixing exactly at a bifurcation, where
  LAD/RCA/LCX legitimately meet, isn't counted as a failure.
  `mean_branch_purity` / `impure_branch_frac` (branches under 95% purity)
  summarize this per image.
- **Cleanliness** -- per class, connected-component count and
  largest-component-size-fraction (`n_components_{class}`,
  `largest_component_ratio_{class}`), plus `spur_count` /
  `spur_density`: short skeleton branches (< `spur_len_px`, default 8) ending
  in a free endpoint -- the standard artifact of a ragged mask boundary after
  skeletonization, not necessarily a real prediction error.
- **Catheter-like FP shapes** -- among false-positive components (predicted
  vessel, GT background), scores each on elongation (major/minor axis ratio),
  skeleton junction count (0 = simple unbranched path), width uniformity
  (catheters are constant-diameter; real vessels taper), and whether they
  touch the image border. `catheter_score` is a tunable heuristic (no GT
  catheter mask exists to validate against), meant to separate "this FP is
  probably the contrast catheter, a known confounder with no class of its
  own" from other false positives.

`segmentation_topology.py`'s component purity and this branch purity are
**not duplicates**: one operates on raw connected-component blobs (whole
predicted shape), the other on the skeleton split at true topological
junctions (individual vessel segments). They can and do disagree, which is
itself informative -- large purity gap between the two usually means the
model produces one blob that's locally clean segment-by-segment but merges
segments that shouldn't be spatially touching.
