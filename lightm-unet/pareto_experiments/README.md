# LM-UNet Pareto Experiments

This folder contains a small runner for short LM-UNet model-size vs Dice sweeps.

The runner varies:

```text
LMUNET_CHANNELS
LMUNET_EDGE_CHANNELS
LMUNET_EPOCHS
LMUNET_BATCH_SIZE
```

through environment variables consumed by `nnUNetTrainerLMUNet`.

## Experiments

The default experiment table is a budget-normalized regional sweep:

```text
B0 baseline
E0 early emphasis
L0 low-mid/stage-2 emphasis
M0 middle/stage-3 emphasis
D0 mid-deep/stage-4 emphasis
P0 stage-5 emphasis
T0 late/stage-6 emphasis
X1 edge-width emphasis
F1 EFE-source emphasis
F2 EFE-source + edge emphasis
```

## Smoke Test

Inside the Docker container:

```bash
cd /workspace/LightM-UNet/lightm-unet
python pareto_experiments/run_lmunet_pareto.py --configs B0 --epochs 1 --batch-size 1 --iters 1 --val-iters 1 --overwrite
```

This only verifies feasibility and timing of a tiny local epoch. It is not a meaningful Dice result.
The runner skips nnU-Net's final full validation by default so this timing reflects the training epoch, not validation export.

## 10-Epoch Sweep

Inside the Docker container:

```bash
cd /workspace/LightM-UNet/lightm-unet
python pareto_experiments/run_lmunet_pareto.py --epochs 10 --batch-size 1 --overwrite
```

By default this uses nnU-Net's planned iterations per epoch unless `--iters` and `--val-iters` are provided.
It also skips final full validation by default. Add `--run-final-validation` only for final candidates.

## HPC Sweep

From the repository root on the cluster:

```bash
sbatch pareto.job
```

The job runs:

```bash
python pareto_experiments/run_lmunet_pareto.py \
  --epochs 10 \
  --batch-size 4 \
  --fold 0 \
  --seed 42 \
  --overwrite
```

The fold is explicitly `0`, not `all`.

## Handoff Follow-Up Sweep

The first sweep suggested that capacity around the conv-to-PV-Mamba transition was most useful. The follow-up runner tests that directly:

```bash
python pareto_experiments/run_lmunet_handoff_followup.py --epochs 10 --batch-size 4 --fold 0 --seed 42 --overwrite
```

Default follow-up configs:

```text
B0   baseline                    12,20,32,44,64,72    edge 20   0.526M
M0   middle_original             16,32,56,56,72,72    edge 20   0.740M
H1   pre_mamba_wide_only         16,32,56,44,72,72    edge 20   0.686M
H2   mamba_entry_wide_only       16,32,44,56,72,72    edge 20   0.697M
H3   smooth_handoff_52           16,32,52,52,72,72    edge 20   0.706M
DS0  downscaled_smooth_handoff   12,24,40,40,56,56    edge 20   0.442M
```

## Outputs

The runner writes:

```text
data/nnUNET_results_pareto/lmunet_pareto/summary.csv
data/nnUNET_results_pareto/lmunet_pareto/pareto_final_dice.png
data/nnUNET_results_pareto/lmunet_pareto/pareto_final_dice_momentum.png
```

Each experiment also gets its own nnU-Net results root under:

```text
data/nnUNET_results_pareto/lmunet_pareto/<ID>_<name>/fold_0/
```

## Metrics

For each run, the parser extracts pseudo Dice from the training log:

```text
final_dice = mean pseudo Dice at the final epoch
best_dice = best mean pseudo Dice across epochs
momentum = final_dice - midpoint_dice
acceleration = late_slope - early_slope
```

For a 1-epoch smoke run, momentum and acceleration are reported as `nan`.
