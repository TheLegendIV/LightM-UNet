# Foundation phase log

Record of what the Foundation phase (`agent_instructions_1.yaml` /
`enet_finn_compression_plan_1.md`) actually did, found, and measured, kept
alongside the code rather than only in chat history. Update this file (don't
replace it) as later stages land significant findings/timings of their own.

## Plan revision: 37-run structured experiment supersedes the original
## stage_1b/stage_2/stage_2b/stage_4 structure in agent_instructions_1.yaml

The user replaced the original generic staged plan with a specific 37-run
structure, built on this session's other findings (U4 as the hardware-
savings-midpoint reference pick, the `f_i==f5` max_unpool constraint, the
params/FLOPs cost-curve work). `agent_instructions_1.yaml` has **not** been
rewritten to match -- this section plus the new `compression/slurm/*.job`
files are the authoritative record of the current plan; the yaml's
stage_1b/stage_2/stage_2b/stage_4 sections (and the corresponding
`stage1b_structure_array.job` / `stage2_grid_array.job` /
`stage2_4_fi_reduction_array.job` / `early_probe_p2_array.job`) are
superseded, not deleted (kept for history).

**Structure** (validated: run counts sum to exactly 37, as specified):

| Section | What | Runs |
|---|---|---|
| 1a | Seed variance (info) on U4, native bottlenecks | 3 |
| 1b | Max-unpool vs. upsample_conv, on U4 with f5 raised to match f_i | 2 |
| 1c | Special-op ablation (dilated/asymmetric/strided/none/DSC) on U4 | 5 |
| 1d | Quantization probe on mid-size model | DEFERRED |
| 2a | Pruning grid: 5 filters (U2/U4/U8/U16/UF) x 4 bottleneck-depth patterns, minus (U4,native) reused from 1a | 19 |
| 2b | f_i fine-tune (16/12/8/4) on 2a's winner | 4 |
| 3a | Quantization (INT16/8/4/2) on the final pruned+f_i-tuned model | 4 |
| 3b | Special-op probe on the final model, for completeness | DEFERRED |

**Real course-correction during scoping, caught by checking the actual grid
definitions rather than assuming:** initially assumed the grid search
required `f_i==f5` symmetry (since E1 has it); checked directly -- only
`E1` does, `U2` through `UF` don't and are valid anyway (only `max_unpool`
requires it). This directly informed 1b's construction: U4's `f5` (8) is
raised to match `f_i` (20) -- `(20,20,36,20,20)` -- the only way to satisfy
`max_unpool`'s requirement while still testing on U4 specifically, not E1.

**New architecture capability added: `use_dsc` (depthwise separable
conv)**, needed for 1c-iii. `RegularBottleneck`'s inner conv (plain or
dilated, not asymmetric -- `use_dsc=True` + `asymmetric=True` raises
`ValueError`, that combination isn't exercised by any planned experiment)
factors into a depthwise `k x k` (groups=internal_channels) + pointwise
`1x1`, standard MobileNet-style factorization. Threaded through `ENet.py`,
`QuantENet.py` (topology parity maintained), both trainers
(`ENET_USE_DSC`), and `collect_results.py` (`--use-dsc`, folded into
`ops_flags`).

**Bottleneck-depth patterns needed for 2a require `bottlenecks_per_stage`
entries of 0** (`variant3`: regular4=regular5=0, "F4 and F5 only upsample
bottlenecks") -- verified this already worked correctly (`nn.Sequential(*[])`
is a valid identity pass-through, no code change needed), both standalone
and through a real truncated training run.

**Added `ENET_SEED`**, needed for 1a: nnU-Net's base trainer only seeds the
train/val split (`np.random.RandomState(12345+fold)`) -- weight init,
augmentation, and dataloader-worker randomness were never explicitly seeded
anywhere upstream. Without this, "3 seed runs" would really have been "3
runs with whatever the ambient unseeded RNG state happened to be" -- not
reproducible or documented. Set as early as possible in `__init__` (before
`.initialize()` builds the network), covering `random`/`numpy`/`torch`/
`torch.cuda`.

**Real bug caught before it reached a real training run**:
`pruning_2b_fi_tune_array.job`'s channel reconstruction treated ENet's
5-value `channels` tuple (`initial, stage1, stage23, stage4, stage5`) as if
it had 6 fields (`f_i,f1,f2,f3,f4,f5`, splitting the shared `f2=f3` slot
into two). `IFS=',' read -r ... <<< "$ENET_CHANNELS"` with 6 variables
against 5 comma-separated values silently left the last variable (`f5`)
empty and shifted every other value over by one -- reconstructed as
`16,20,36,8,` (trailing comma, `f4` got what should've been `f5`'s value,
`f5` empty). Caught by reproducing the exact `read` command standalone
before trusting it, not by inspection. Fixed to 5 variables
(`_BEST_FI BEST_F1 BEST_F23 BEST_F4 BEST_F5`).

**Config files**: `compression/configs/cfinal_ops.env` (decoder_type + op
flags incl. `ENET_USE_DSC`, sourced by 2a/2b/3a) repurposed from the old
plan's stage_1b output to the new plan's 1b/1c output -- same mechanism,
updated docs. New `compression/configs/best_model.env` (2a/2b's winning
channels/bottlenecks, sourced by 2b/3a), placeholder-defaulted to U4 x
native until 2a/2b actually run.

**Smoke testing** (per the instruction to smoke test each and report):
- All **37 configs** individually constructed and forward-passed at
  512x512 (exhaustive, not sampled) -- one Python script replicating each
  job's exact bash array-index logic, to catch config-generation bugs
  before they'd reach real training. All 37 passed.
- Two real truncated `nnUNetv2_train` runs (2 epochs, checkpointed, full
  200-case validation) for the two genuinely new capabilities exercised
  through the actual pipeline, not just construct/forward: `1c_none_dsc`
  (DSC across every bottleneck) and `2a_UF_var3` (regular4/regular5 both
  empty). Both completed cleanly -- checkpoints, debug.json, progress.png,
  validation outputs all written. Test debris cleaned from
  `data/nnUNet_results/` afterward (not gitignored, see earlier finding).
- Did not re-smoke-test the general FP32/QAT pipeline itself (already
  verified end-to-end earlier this session, see the QAT trainer section
  above) -- only the two things that are actually new.

**Not yet done**: 1d and 3b are deferred per the plan, not built.
`agent_instructions_1.yaml` itself not updated to match this structure --
flagged to the user as a follow-up decision, not done unilaterally.

## Plan revision 2: re-baselined from E1 to Original, U4 -> O4

Real checkpoints for both Stage-1 baselines (`E1`, `Original`) landed
mid-session, run through the (already-fixed-this-session)
`collect_results.py` inference path -- and the real test-set numbers
flipped the plan's foundation: **`Original` beats `E1` on Dice (0.7800 vs
0.7746) while already being smaller** (369,497 vs 466,294 params, 4.80B vs
6.08B MACs). Since the entire 37-run structure's filter axis (U2/U4/U8/U16/UF)
and reference pick (U4) were geometric scale-downs of *E1's* specific
channels (20,72,144,72,20), not Original's (16,64,128,64,16), this
invalidated the foundation, not just one number -- confirmed with the user
(re-derive the whole axis from Original, and rebase section 1 onto the new
reference) before rebuilding anything.

**Three more real bugs, all caught only once real checkpoints existed --
none of the earlier construct/forward-pass smoke tests could have caught
any of them, since they only manifest during actual checkpoint loading /
real training history inspection:**

1. **Checkpoint folder structure mismatch.** The user's real checkpoints
   arrived flat under `nnUNet_results/{trainer}__{plans}__{config}/`,
   missing the `Dataset501_ARCADE/` dataset-name subfolder every job script
   and `collect_results.py` assumes (nnU-Net's own convention). Most likely
   an `scp`/copy step that flattened one directory level. Fixed by moving
   the three folders into the expected location -- a local reorganization,
   not a data change.
2. **`regular5` checkpoint incompatibility.** `E1`/`Original` were trained
   on the pre-parametric `ENet.py` (before this session's
   `bottlenecks_per_stage` work), where `regular5` was a single bare
   `RegularBottleneck` (`regular5.reduce...`), not an `nn.Sequential`
   (`regular5.0.reduce...`). `regular1`/`regular4` were already
   `Sequential`-wrapped before this session, so only `regular5` actually
   changed shape. Fixed with a backward-compatible `ENet.load_state_dict`
   override that migrates legacy keys (`regular5.X` -> `regular5.0.X`) --
   a pure rename, not an architecture change, since both checkpoints have
   exactly 1 `regular5` rep. Verified against the real `E1` checkpoint
   before trusting it.
3. **`collect_results.py` had the exact same 5-vs-6-field channel-unpacking
   bug** (conflating `f2=f3`'s shared slot with two separate fields) found
   and fixed in `pruning_2b_fi_tune_array.job` earlier this session --
   except this instance had never actually been exercised until real
   inference ran end-to-end for the first time. `f_i, f1, f2, f3, f4, f5 =
   args.channels` against a 5-tuple crashed immediately. Fixed the same way
   (unpack 5, assign the shared value to both `f2`/`f3` result columns).

**Also found: `read_training_info`'s `epochs`/`converged_flag` were reading
the wrong file.** `debug.json` turned out (checked against a real
checkpoint, not assumed) to be a static snapshot written once at trainer
initialization -- it showed `current_epoch=0` for a fully-trained 150-epoch
run, silently wrong for every real run, not just an edge case. Real epoch
count lives inside the checkpoint itself (`checkpoint_best.pth`'s
`current_epoch=130` for `E1`, i.e. best validation performance occurred at
epoch 130 of 150). Rewrote `read_training_info` to read `current_epoch`
from the checkpoint being used for inference, and to compute
`converged_flag` properly (per the yaml's actual definition, "still rising
at the end") from `checkpoint_final.pth`'s full `ema_fg_dice` logging
history (mean of the trailing-15-epoch window vs. the window before it) --
not the previous "did it reach num_epochs" heuristic, which conflated
"completed all epochs" with "had plateaued." Real result: `Original`'s
`converged_flag=False` (still improving at epoch 148) -- a genuinely useful
finding (Original might benefit from more epochs) that the old heuristic
couldn't have produced.

**Notebook execution attempted, then explicitly stopped by the user.**
Installed `nbconvert`/`ipykernel`/`jupyter_client` (added to
`requirements-enet-base.txt`), found and fixed the *same*
`nnUNetv2_predict` (bare trainer class) bug in
`analysis/501_ARCADE/preview_results.ipynb` that `collect_results.py` had
(same root cause: folder resolution can't distinguish `nnUNetTrainerENet_E1`
from `nnUNetTrainerENet_Original`, both sharing one trainer class) --
switched it to `nnUNetv2_predict_from_modelfolder` the same way. Execution
via `jupyter nbconvert --execute` got through inference successfully but
hit an unrelated pre-existing bug (`KeyError: 'territory'` in a topology
cell expecting per-branch columns `evaluate_case` doesn't produce) before
the user said to stop pursuing it and rework the experiments instead. The
inference-bug fix is committed; the notebook was never fully executed for
either model, and the `territory` KeyError is unfixed.

**Re-derivation, same methodology throughout (not a fresh judgment call
each time), all confirmed against real numbers:**
- New filter axis, `f_i=16` fixed (was 20): `Original`=(16,64,128,64,16),
  `O2`=(16,32,64,32,8), `O4`=(16,16,32,16,4), `O8`=(16,8,16,8,4),
  `O16`=(16,4,8,4,4), `OF`=(16,4,4,4,4).
- New reference pick: **`O4`**, found the same way `U4` was (closest
  geometric-axis point to the sub-range savings midpoint, not the naive
  full-range midpoint) -- `O4`=76.76% savings vs. the `O2`-`OF` midpoint of
  71.39% (5.4pts off), `O8`=82.58% (11.2pts off, clearly worse). Same
  relative axis position as `U4` was, same diminishing-returns shape.
- Section 1 (1a/1b/1c) rebased onto `O4`/`O4_sym`=(16,16,32,16,16) (the
  `f_i==f5` symmetric variant needed for 1b's max-unpool comparison, same
  construction as `U4_sym` before).
- Section 2a's grid: same 5x4 structure (5 filters x 4 bottleneck
  patterns, minus the now-`(O4,native)` cell reused from 1a), same 19 tasks.
- Section 2b's f_i sweep shrank from 4 tasks (`[16,12,8,4]`) to 3
  (`[12,8,4]`) -- `f_i=16` is already closer to the floor than E1's `f_i=20`
  was, so the same -4-per-step sweep terminates one step sooner. A real,
  non-arbitrary consequence of the re-baseline, not copied over blindly.
- All new configs (`O4`, `O4_sym`, all 19 `2a` grid cells, the 3 new `2b`
  f_i values) individually construct+forward-pass verified before touching
  any job script.
- `compression/configs/best_model.env`/`cfinal_ops.env` defaults and
  comments updated to `O4`/Original throughout.
- `compression/generate_cost_tables.py` re-baselined and rerun (Original,
  not E1) -- `plot_cost_relationships.py` did NOT need rerunning, it's
  anchored to floor channels `(4,4,4,4,4)`, not any specific baseline,
  confirmed by inspection before assuming either way.
- `compression/generate_hardware_savings_ranking.py` rewritten for the new
  `2a` grid/naming convention and rerun -- and its Original-baseline Dice
  now uses the REAL measured value (0.7800019869577919) instead of the
  user-supplied 0.83 placeholder from earlier, since it exists now. Sanity
  check re-passed: `2a_Original_native` scores exactly `c=0.333`, the
  theoretical baseline point.
- `compression/generate_symmetric_reduction_family.py` (the fine-grained
  `E1_fixfi_k*` interpolation used earlier to sanity-check `U4`): initially
  left un-rerun since `O4` was found directly via the geometric-axis method
  this time, no interpolation needed -- the user flagged this as an
  inconsistency (it's still part of "the cost analysis"), so it was
  rebaselined and rerun too. `Original_fixfi_k4` = `(16,32,96,32,4)` is the
  closest interpolated point to 50% savings (47.52%, 17 points k=0..16,
  converges exactly to `OF` -- both sanity checks re-passed).

## What was built

- `enet/nnunetv2/nets/ENet.py`: fully parametric (`bottlenecks_per_stage`,
  `decoder_type`, `use_dilated`/`use_asymmetric`/`use_strided`, `max(1,·)`
  clamp on internal channels). Self-test in `__main__` builds + forward-passes
  112 config combinations at 512x512.
- `enet/nnunetv2/training/nnUNetTrainer/nnUNetTrainerENet.py`: new env vars
  (`ENET_BOTTLENECKS`, `ENET_DECODER_TYPE`, `ENET_USE_DILATED`,
  `ENET_USE_ASYMMETRIC`, `ENET_USE_STRIDED`) wired into
  `build_network_architecture`, same pattern as the existing `ENET_CHANNELS`.
- Brevitas 0.12.1 added to `setup-enet.sh` / `requirements-enet-base.txt`.
- `compression/utils.py`: `count_params`, `count_flops` (extracted from
  `analysis/501_ARCADE/record_architecture_stats.py`, which now imports
  them instead of duplicating), `count_bops` stubbed for Stage 4.
- `compression/collect_results.py`: runs inference if needed, computes
  dice/clDice/n_components (via `analysis/501_ARCADE/segmentation_topology.py`)
  + params/FLOPs, writes/updates a `results.csv` row.
- `analysis/501_ARCADE/segmentation_topology.py`: added `cldice_score`
  (centerline Dice), reusing the existing `skeletonize` import rather than a
  second skeletonization implementation.
- `compression/smoke_test.sh`: pre-flight smoke test wrapper (env-var driven,
  no new data-subsetting code).
- `compression/notebook/report.ipynb`: single reporting point, one section
  per stage, reads `compression/results.csv`.
- `compression/generate_cost_tables.py`: Stage 2.2's architecture-only
  marginal-cost tables (no training) -- generated, see below.
- `enet/nnunetv2/nets/QuantENet.py`: homogeneous-bit-width Brevitas-quantized
  mirror of `ENet.py` (same constructor + bit-width knobs), for Stage 4 and
  the P1 FINN probe. `compression/finn_resource_probe.py`: exports a
  QuantENet config to QONNX and (if available) pushes it through FINN's
  analytical estimate step. `qonnx`/`onnx`/`onnxscript`/`onnxoptimizer`
  added to `requirements-enet-base.txt`/`setup-enet.sh` alongside Brevitas.

- `compression/slurm/stage1b_structure_array.job`: Stage 1b's 5-task array
  (2 decoder + 3 op-ablation, all off the Stage-1 `E1` baseline). Skips
  training if a checkpoint already exists for that task, then always runs
  `collect_results.py`.
- `compression/plot_cost_relationships.py`: config-independent cost curves,
  both params AND FLOPs (per filter, per block, vs. current width) --
  `compression/cost_tables/cost_relationships.png` (params) +
  `cost_relationships_flops.png` (FLOPs) + the underlying CSVs. FLOPs and
  params disagree on which stage is priciest: `stage23` (widest channels)
  for params, but `f5` (highest resolution -- 1/2, vs. `stage23`'s 1/8) for
  FLOPs -- optimizing for one doesn't optimize for the other.
- `compression/slurm/stage2_grid_array.job` (24 tasks) and
  `stage2_4_fi_reduction_array.job` (8 tasks), both sourcing
  `compression/configs/cfinal_ops.env` (Stage 1b's not-yet-decided
  decoder_type/op-flags output) so that gets edited in one place once
  Stage 1b actually runs.
- `compression/plot_sweep_progress.py <stage>`: generic Dice-vs-params/FLOPs
  progress PNG per sweep, safe to re-run mid-sweep.
- `compression/README.md`: folder map + how to run a sweep.
- `generate_cost_tables.py` gained a third table/plot: activation/feature-map
  memory per stage (`activation_memory.csv`/`.png`) -- see below.
- `enet/nnunetv2/training/nnUNetTrainer/nnUNetTrainerENetQuant.py`: the QAT
  trainer wired to `QuantENet` -- see below, built and verified end-to-end.

## Findings

**1. Real bug, caught by the self-test, not written into spec beforehand:**
`DownsamplingBottleneck`'s main branch only handled the channel count
*growing* (zero-pad). Several Stage 2 filter configs (U8/U16/UF) have
`stage1_channels < f_i` -- the main branch needs to *shrink* there, which
didn't exist as a code path before (the two pre-existing baselines,
`enet_paper`/`E1`, both only expand at every downsampling step). Fixed: the
main branch now truncates to the target channel count when it needs to
shrink, keeping the branch parameter-free either direction.

**2. Architectural constraint (real, not a bug) -- decoder/channel symmetry:**
`max_unpool` decoder requires `MaxUnpool2d`'s indices to match the decoder
stage's channel count exactly: `initial_channels == stage5_channels` and
`stage1_channels == stage4_channels`. True for `E1`/`enet_paper` (both
symmetric configs) but **not** for the Stage 2 filter axis (U2/U4/U8/U16/UF
all have `f5 != f_i`) -- that axis is only valid under `upsample_conv`. This
is the concrete mechanism behind why Stage 1b has to lock the decoder before
Stage 2's grid runs, not just an ordering preference.

**3. Real bug, found while writing the Stage 1b job script --
`collect_results.py`'s inference step was looking in the wrong folder for
every config except one.** `nnUNetv2_predict`'s folder resolution
(`get_output_folder`) depends only on `-tr/-p/-c` (trainer class name,
identical across every sweep config: always `nnUNetTrainerENet`) -- it
can't distinguish `nnUNetTrainerENet_E1` from
`nnUNetTrainerENet_stage1b_no_dilated`, both trained via the same trainer
class with `ENET_OUTPUT_FOLDER` redirecting each to its own
`net_name`-suffixed folder (the same convention `record_architecture_stats.py`
and `train_enet_e1.job` already used correctly). Fixed: switched to
`nnUNetv2_predict_from_modelfolder` (`-m <exact folder>`), which sidesteps
the ambiguity entirely. Would have silently looked in (or written to) the
wrong checkpoint's folder the first time two configs shared a trainer class
-- i.e. on literally the first real sweep run past Stage 1.

## Analytical cost tables (Stage 2.2, architecture-only, no training)

`compression/cost_tables/filter_cost.csv` / `bottleneck_cost.csv`, baseline
= E1, `upsample_conv`. Two things worth noting for the write-up (both
contradicted an initial guess written into the generator script's comments,
corrected after seeing the real numbers, not left as speculation):

- **f_i is the *cheapest* filter slot to grow** (+4 filters: 336 params /
  16.8M FLOPs), not the most FLOP-heavy as might be assumed from it feeding
  the highest-resolution stage -- it only feeds the initial block + down1's
  main branch, nothing downstream scales with it directly.
- **stage2 and stage3 cost identically** per bottleneck/filter added (both
  run at the same 1/8 resolution, between down2 and up4) -- there's no
  "narrower/cheaper" one between them, contrary to an initial assumption.
- The shared stage2/3 width (`f2=f3` in the yaml's per-row tables -- one
  actual knob in `ENet.py`'s 5-value `channels` tuple, not two independent
  ones) is the single most expensive filter slot to grow: +4 filters costs
  22,651 params / 462M FLOPs, ~14x f_i's cost for the same +4 step.

## Activation / feature-map memory axis (third axis beyond params/FLOPs)

`compression/cost_tables/activation_memory.csv` + `.png` --
`generate_cost_tables.py`'s `main_activation_memory()`, verified via forward
hooks on a real `ENet` instance (real tensor shapes, not stride arithmetic
that could silently drift from the actual downsampling schedule). At
512x512 input: `f_i` and `f5` are TIED for the largest feature maps (both
256x256 = 65,536 elements/channel -- `InitialBlock` does its own stride-2
downsample immediately, so `f_i` is at H/2, not the full input resolution),
`f1`/`f4` are 4x smaller (128x128 = 16,384), and `stage23` is 16x smaller
still (64x64 = 4,096). This inverts the params story: `f_i` is the
*cheapest* stage in params (16/filter, see below) but ties for the
*most expensive* in activation memory -- relevant for ZU7EV since on-chip
buffering (BRAM) is often the real constraint, not weight storage. Memory
per channel is exactly linear (no block-count dependence, unlike
params/FLOPs) -- total activation memory for a stage = elements_per_channel
x channel_width x bytes/element (bytes/element depends on Stage 4's
quant_bits).

`f_i`'s exact params-per-filter rate (16, constant at every width -- see
`normalized_filter_cost.csv`) has a clean closed-form derivation: `f_i` only
feeds `InitialBlock`'s conv (marginal 9/channel: `9x(C-1)` weight, kernel
3x3) + its BatchNorm2d (marginal 2/channel) + its PReLU (marginal
1/channel) + `down1`'s reduce conv (marginal 4/channel: `Cx1x2x2`, since
`down1`'s internal_channels stays fixed at floor=1 throughout, independent
of `f_i`). 9+2+1+4 = 16, exactly matching the measured rate -- confirms
it's genuinely linear (not just "small enough to look flat" over the
sampled range), because `f_i` is the one slot that doesn't feed into a
squared-both-sides bottleneck block.

## FINN resource probe (P1) -- in progress, half done

Real, verified progress, not yet the full deliverable:
- `QuantENet` (U8 @ INT8, and E1/UF at other bit-widths) builds, forward-passes,
  and passes a topology-parity check against `ENet.py` (same per-stage depths).
- `brevitas.export.export_qonnx` succeeds and the output passes `onnx.checker`
  (`compression/configs/early_probes/U8_int8.onnx`).
- **Not yet done**: the actual FINN analytical estimate
  (`DataflowOutputType.ESTIMATE_REPORTS`) -- confirmed this doesn't need
  Vivado/Vitis, but FINN only officially runs inside its own dedicated
  Docker container (github.com/Xilinx/finn), separate from both Vivado/Vitis
  and this repo's training container. Not set up. Re-run
  `compression/finn_resource_probe.py` once it is -- the script already
  detects FINN's absence and stops cleanly rather than failing silently.
- Deviation to account for in Stage 4 write-up: `QuantENet` uses `QuantReLU`
  instead of ENet's `PReLU` throughout (no standard quantized/FINN-dataflow
  PReLU exists in Brevitas) -- Stage 4 Dice numbers on `QuantENet` won't be
  directly comparable to the FP32 baselines without accounting for this
  activation-function change, not just the precision change.

## QAT trainer (`nnUNetTrainerENetQuant`) -- built and verified end-to-end

`enet/nnunetv2/training/nnUNetTrainer/nnUNetTrainerENetQuant.py`: same
env-var pattern as `nnUNetTrainerENet` (inherits it directly, reuses its
`_parse_channels`/`_parse_bottlenecks`/`_parse_bool_env` helpers), plus
`ENET_QUANT_BITS` (homogeneous weight+activation bit-width, applies to both
-- heterogeneous per-layer, Stage 4.2, isn't built). Builds `QuantENet`
instead of `ENet`. Rejects `ENET_QUANT_BITS>=32` -- not a meaningful QAT
config; the FP32 reference point is whatever Stage 1/2 already trained with
plain `nnUNetTrainerENet`.

Found and fixed one bug getting here: `QuantENet.forward` could return a
Brevitas `QuantTensor` instead of a plain `torch.Tensor` (from the final
`QuantConvTranspose2d`), which nnU-Net's loss functions don't understand.
Fixed by unwrapping `.value` before returning.

Verified in two steps, each a stronger check than the last:
1. Standalone forward+backward through `QuantENet` (UF, INT8) using the
   real `DC_and_CE_loss` on a GPU tensor -- loss computes, every parameter
   gets a gradient.
2. A real `nnUNetv2_train` smoke test (UF, INT8, 3 truncated epochs) via
   `nnUNetTrainerENetQuant` -- trained, checkpointed, and ran full
   validation over all 200 test cases (mean Dice 0.070 -- meaningless at 3
   epochs on the floor config, the point was pipeline correctness, not the
   number).

Together these mean early-probe P2 and Stage 4's homogeneous sweep (4.1)
can run today, on real HPC, via
`nnUNetv2_train Dataset501_ARCADE 2d 0 -tr nnUNetTrainerENetQuant` with
`ENET_QUANT_BITS` set alongside the usual `ENET_CHANNELS` etc.
`compression/slurm/early_probe_p2_array.job` -- 3 tasks (E1 at INT8/INT4/INT2,
`bits=32` reuses Stage 1's existing FP32 `E1` row instead of retraining it)
-- is now built.

**Second real bug, caught before the sweep ran, not after:** `count_bops`
was a stub, and `collect_results.py` always built a plain FP32 `ENet` for
params/FLOPs regardless of `--quant-bits`. Fixing both surfaced a third bug:
`thop` (via `count_flops`) silently **undercounts a `QuantENet`'s FLOPs by
~40x** (307M vs. the FP32-equivalent's 12.16B, measured on E1) -- it
doesn't recognize Brevitas's `QuantConv2d`/`QuantConvTranspose2d` as
countable ops, so most of the network contributes ~0 to its profile. Fixed
by computing MACs from the topology-equivalent FP32 `ENet` (verified
topology-identical to `QuantENet`, and MAC count is determined by topology,
not bit-width) rather than trying to get `thop` to understand Brevitas
layers, then `count_bops(macs, bits)` = `macs x weight_bits x act_bits`
(the standard BOPs convention -- a b_w-by-b_a-bit MAC costs roughly
proportional to the product of the two bit-widths, not their sum).
`collect_results.py` now builds the FP32 model for FLOPs/MACs and the real
`QuantENet` for params (close to but not identical to the FP32 count --
~462k vs. 466k on E1, from Brevitas's per-layer scale-factor bookkeeping)
when `--quant-bits != 32`, and passes `ENET_QUANT_BITS` through to
inference (recovered correctly at predict time via the checkpoint's own
stored `trainer_name`, not a CLI flag -- confirmed by reading
`nnUNetPredictor.initialize_from_trained_model_folder`'s source rather than
assuming). Verified: BOPs scales exactly as `bits^2` across INT8/4/2
(16:4:1), params stays constant across bit-widths (correct -- quantization
doesn't change tensor shapes), `params_x_bits` scales linearly.

## Hardware-savings ranking (Stage 2 grid vs. E1)

`compression/generate_hardware_savings_ranking.py` -> `cost_tables/hardware_savings_ranking.csv`/`.png`.
User-specified minimization score: `alpha*MACs_ratio + beta*memory_ratio -
c*Dice_ratio` (alpha=beta=c=1/3; Dice term SUBTRACTED, not added as
literally first written -- confirmed with the user, since "+" would have
made better accuracy increase a score meant to be minimized). Memory ratio
combines activation-elements + params at an assumed 8-bit width that
cancels exactly in the ratio (same B on both sides), computed from the
per-stage activation-element constants already verified in
`generate_cost_tables.py`.

**Important caveat, not silently glossed over**: Stage 2 hasn't trained yet
(no `results.csv`), so all 24 rows currently use a Dice PLACEHOLDER
(assumed parity with E1) -- the ranking right now reflects pure hardware
cost only, not a real accuracy/hardware tradeoff. The script reads real
Dice from `results.csv` automatically when present (falls back to the
placeholder per-row, flagged via `dice_is_placeholder`) -- re-run as-is
after Stage 2 actually trains, no changes needed. Sanity check:
`stage2_E1_bnnative` (the grid cell identical to E1 itself) scores exactly
`c=0.333`, the theoretical baseline point (`macs_ratio=mem_ratio=dice_ratio=1`).

## Reference pick: U4, chosen as the mid-savings point on the filter axis

**Decision: `U4` (channels `20,20,36,20,8`, bnnative, `upsample_conv`) is the
reference pick for "roughly half of achievable hardware savings", not
`U8`** (the filter axis's index midpoint) and not a bespoke intermediate
config. Process that led here, in order:

**1. "Which grid config gets 80% of UF's savings?"** None does exactly --
`U2/bn2` (73.8% of UF's savings) and `U4/bn2` (88.6%) bracket it, `U2/bn2`
closer. Flagged that bottleneck depth barely moves this metric (<0.5pt
spread within any filter family) -- filter width does essentially all the
work.

**2. "Which grid config gets the 0%(E1)-to-100%(UF) midpoint?"** Also none
exactly -- `U2/bnnative` (65.8%) closest, `E1/bn2` (31.1%) next. Revealed a
real structural gap: the filter axis is geometric (each step halves width),
so E1->U2 is the single largest step in the whole grid -- nothing samples
between "full width" and "half width."

**3. Built a continuous E1-structure-preserving reduction family to probe
that gap** (`compression/generate_symmetric_reduction_family.py`), with
real course-correction along the way, each caught by checking actual
numbers rather than assuming:
   - First attempt fixed `f_i` AND `f5` (assumed the grid search required
     `f_i==f5` symmetry throughout, since E1 has it). Wrong -- checked the
     actual grid definitions directly: only `E1` has `f_i==f5` (inherited
     from being the paper-faithful baseline), `U2` through `UF` all break
     it and are valid anyway, since Stage 2 uses `upsample_conv`, and only
     `max_unpool` requires that symmetry (foundation_log.md finding #2).
   - Corrected to fix `f_i` only, per the user's actual instruction. Fixed-`f5`-too
     literal-reduction-by-8 caps at 28.3% savings before `f5` (which starts
     at 20, same magnitude as `f_i`) goes negative -- nowhere near 50%,
     shown explicitly rather than silently working around it.
   - Final construction: fix `f_i` only; reduce `f1`(=`f4`), `f2`=`f3`, and
     `f5` each by 8 per step, independently floored at 4 (matches the grid's
     own floor). Two internal consistency checks both passed: `E1` (k=0)
     gives exactly 0% savings, and the scheme's final point (k=18, every
     stage floored) is bit-for-bit identical to `UF` -- confirms this
     construction is a genuine interpolation between the two, not an
     unrelated family.

**4. "Is U4 really the visual midpoint between U2 and UF, or should U8 (the
axis's integer midpoint) be used instead?"** Computed savings for the full
filter axis (bnnative, `alpha=beta=0.5`, no accuracy term):

| Config | Savings |
|---|---|
| E1 | 0.00% |
| U2 | 56.18% |
| **U4** | **73.88%** |
| U8 | 81.42% |
| U16 | 83.54% |
| UF | 85.34% |

Midpoint of U2 and UF's savings = 70.76%. **U4 sits 3.12 points from it;
U8 sits 10.65 points from it** (U8 is already 86.5% of the way from U2 to
UF, not a midpoint). Confirmed by the marginal steps: U2->U4 buys +17.7pts
of savings, U4->U8 only +7.5, U8->U16 +2.1, U16->UF +1.8 -- strongly
diminishing returns past U4. The filter axis's *index* midpoint (U8) and
its *savings* midpoint (U4) diverge because the params/FLOPs-vs-width
relationship isn't linear (see the params/FLOPs cost-curve findings above)
and because `f_i`/`f5` stay fixed-cost, resolution-driven contributors that
don't shrink with the filter axis at all -- so most of the achievable
savings are already captured by U4, and pushing further (U8/U16/UF) trades
a lot more accuracy risk for comparatively little additional hardware
saving.

**Caveat carried over from every hardware-only analysis above: none of this
uses real Dice yet** (Stage 2 hasn't trained) -- U4 is justified purely as
the hardware-savings midpoint, not yet validated as meeting the Dice goals.
That validation happens once Stage 2 actually trains U4 (and its
bottleneck-depth siblings) and `collect_results.py` populates real numbers.

## Training-time estimates

- **Local docker (`pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel`) smoke test**:
  `UF` (floor config, 1,322 params, ~97M FLOPs), `upsample_conv`, full
  iteration counts (250 train / 50 val), default batch size. Steady-state
  epoch time **~107s** -> ~4.5h/150-epoch run. This GPU is weak relative to
  HPC (see below) -- useful only for confirming the pipeline runs correctly,
  **not** for capacity planning.
- **HPC `gpu_mig`, authoritative**: `E1` (466,294 params, ~12.16 GFLOPs,
  the largest config in the near-term plan), 150 epochs = **80 minutes**
  (~32s/epoch) -- user-measured, real. Use this number for scheduling, not
  the local-docker one.

## Total planned runs (near-term: Stage 1b through Stage 2.4)

| Batch | Runs |
|---|---|
| Stage 1b (2 decoder + 3 op-ablation) | 5 |
| Early probe P2 (bits 32/8/4/2, decoupled from main path) | 4 |
| Stage 2 grid (6 filter x 4 bottleneck) | 24 |
| Stage 2.4 (f_i in [20,12,8,4] on 2 configs) | 8 |
| **Total (gross)** | **41** |
| after obvious dedup (E1 grid cell, f_i=20 rows) | ~38 |

At 80 min/run (E1's measured HPC time, used as a conservative uniform
proxy -- smaller configs are likely faster): **41 runs x 80min ~= 54.7
GPU-hours serial**. On `gpu_mig` as a Slurm array (1 GPU/job, parallel
across however many MIG slices are available concurrently), wall-clock is
roughly that total divided by the number of concurrent slices.

Stage 4 quantization (homogeneous 4 + heterogeneous ~6) adds ~10 more runs
when reached, bringing the full near-term + Stage-4 total to ~51 runs
(~48 after dedup), ~68 GPU-hours serial at the same 80min/run proxy.

Deferred/optional (not included above, not run without a cue): Stage 2.5
(3 variants), Stage 2.6 (contingency, variable). Stage 2b scaled down from
the original 3-configs-x-3-seeds (6-9 runs) to **2 additional seeds on
`Cfinal_arch` only** (3 total incl. its grid run) -- scope is producing the
best deployable model, not a general ENet sensitivity study, so confirming
the one selected architecture isn't a lucky outlier is enough; a broader
noise-floor study across multiple configs was judged excessive. Stage 3
merged into Stage 1b (see above). Stage 5 (full FINN folding/synthesis)
out of scope for now -- its analytical-estimate half is P1, in progress.

## Open question worth testing cheaply

Local-docker UF (tiny, 1,322 params) took longer per epoch (~107s) than
HPC E1 (466k params, ~32s/epoch) -- suggests HPC throughput here may be
data/augmentation-bound rather than compute-bound, which would mean smaller
Stage 2 configs won't run proportionally faster than E1 on HPC. Not
confirmed: a single UF run on HPC (<=80min) would settle it before trusting
the 80min/run uniform proxy across the whole grid.
