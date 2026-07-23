# upscale/ -- pre-compression baseline-boosting study

Runs **before** the 37-run compression pipeline in `compression/`. Where
`compression/` prunes *down* from `Original` (16,64,128,64,16 / bn 4,8,8,2,1),
this asks the opposite question: does scaling ENet *up* from `Original` find
a better real-Dice baseline to prune from instead?

## Methodology (cherry-picked from `wip_pc`)

The `wip_pc` branch has an earlier, unrelated "dirty" optimization study:
short (15-epoch) pareto sweeps over hand-designed architecture variants,
plotted as params-vs-Dice scatter, winners picked off the front by eye, then
re-run ("graduated") at full epoch count. That harness -- the sweep runner,
training-log scraping, CSV/plot output, graduate-the-winners step -- is what
got ported here. Its actual *architectures* (`ENetUpscaled`,
`ENetUpscaleArch`, all new `nn.Module` classes) were **not** ported: this
repo's `enet/nnunetv2/nets/ENet.py` is already fully parametric (channels,
`bottlenecks_per_stage`, decoder type), so every experiment is expressed
through those existing knobs instead, reusing `nnUNetTrainerENet`'s `ENET_*`
env vars unchanged.

Op flags (dilated/asymmetric/strided/DSC) are held fixed at `Cfinal_ops`
(`compression/configs/cfinal_ops.env`'s choices) throughout -- this sweep
only varies *capacity* (channel width, bottleneck depth), not the topology
decisions already locked in by stage 1b.

## The 3 tracks (45 configs)

Combinatorial over "which stage(s) get bumped": each stage independently,
then 2 together (widths stop there; depth stops there too -- "less is fine
if sufficient"). Two axes -- channel **width** and bottleneck **depth** --
split across three tracks because `max_unpool` constrains the width axis:
`MaxUnpool2d` needs `down1`/`down2`'s pooling INPUT widths (`initial_channels`,
`stage1_channels`) to match `up5`/`up4`'s projected OUTPUT widths
(`stage5_channels`, `stage4_channels`), forcing `f_i==f5` and `f1==f4`.
`stage2`/`stage3` have no pooling of their own, so they're unaffected by
that constraint -- splitting them into independent widths (`ENet.py`'s new
6-tuple channels form, `(f_i, f1, f2, f3, f4, f5)`) recovers a 4th DoF under
`max_unpool`. Depth (`bottlenecks_per_stage`) has no bearing on the
constraint at all, so it doesn't need a decoder split.

| Track | IDs | Axis | Slots | Decoder | Combos | Bump |
|---|---|---|---|---|---|---|
| A | `MU01`-`MU15` | channel width | stem(`f_i`=`f5`), flank(`f1`=`f4`), `f2`, `f3` | `max_unpool` | size 1-4 (`C(4,1..4)`) | 1.75x |
| B | `UC01`-`UC15` | channel width | `f_i`, `f1`, `f23`, `f4`, `f5` | `upsample_conv` | size 1-2 (`C(5,1..2)`) | 1.75x |
| C | `BD01`-`BD15` | bottleneck depth | `stage1`, `stage2`, `stage3`, `regular4`, `regular5` | `upsample_conv` | size 1-2 (`C(5,1..2)`) | +4 blocks |

Worst-case params: ~1.11M (track A/B) and ~516K (track C) -- both under the
~1.1M relaxed cap for this sweep. See `run_upscale_pareto.py`'s docstring
for the exact derivation and per-config hypotheses.

## Files

- `pareto_common.py` -- generic sweep harness (training-log dice-curve
  parsing, momentum/acceleration proxies, CSV/plot helpers). Trainer- and
  experiment-table-agnostic.
- `run_upscale_pareto.py` -- builds the 45-config `EXPERIMENTS` table
  (3 tracks, see above) and runs the 15-epoch sweep.
- `graduate.py` -- re-runs chosen winners at full epoch count (default 150)
  with checkpointing + final validation back on, then calls
  `compression/collect_results.py` so the graduated result lands in
  `compression/results.csv` (`stage=upscale_graduate`) next to
  `Original`/`O2`/`O4`/... for direct comparison.
- `slurm/upscale_pareto_array.job` -- 45-task array, one per `MU*`/`UC*`/`BD*`
  (IDs read from `run_upscale_pareto.py` itself, not hardcoded in the job).
- `slurm/upscale_graduate_array.job` -- array over manually-picked winner
  IDs (edit `WINNER_IDS` before submitting).
- `results/pareto_e15/` -- sweep output (`summary.csv`, `pareto_final_dice.png`,
  per-config subfolders). Throwaway/dirty by design -- not meant to be kept
  long-term the way `compression/results.csv` is.

## Usage

```bash
# sanity-check the experiment table + param counts, no training
python upscale/run_upscale_pareto.py --dry-run

# full 15-epoch sweep (or a subset)
python upscale/run_upscale_pareto.py
python upscale/run_upscale_pareto.py --configs MU08 UC03 BD10

# inspect results/pareto_e15/summary.csv + pareto_final_dice.png, pick winners by eye

# graduate winners to 150 epochs, auto-collect into compression/results.csv
python upscale/graduate.py --configs MU08 BD10
```

On HPC: submit `slurm/upscale_pareto_array.job`, inspect results, edit
`WINNER_IDS` in `slurm/upscale_graduate_array.job` (and its `--array` range
to match the winner count), then submit that.

## What happens if a graduated config wins

If a graduated `upscale_*` row beats `Original` in `compression/results.csv`,
that becomes the new starting point for the compression pipeline: rebase the
filter/hardware-savings axis onto it the same way `Original` itself replaced
`E1` earlier in this project (see `compression/foundation_log.md`'s
"Plan revision 2" for that precedent) before re-deriving the reference pick
and re-submitting `compression/slurm/*.job`.
