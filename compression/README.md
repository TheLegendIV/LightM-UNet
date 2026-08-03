# compression/

Sweep orchestration, analysis, and reporting for the ENet compression
experiment (`agent_instructions_1.yaml` / `enet_finn_compression_plan_1.md`).
Everything *new* this experiment needs lives here; core model/trainer code
stays under `enet/nnunetv2/` (nnU-Net's `-tr` lookup only finds trainer
classes there) and existing dice/topology analysis stays under
`analysis/501_ARCADE/` -- this folder calls into both rather than
duplicating them. See `foundation_log.md` for the running list of findings,
bugs caught, and timing measurements.

## Layout

- `results.csv` -- unified per-run table, one row per config
  (`collect_results.py` writes it). Doesn't exist until the first run lands.
- `utils.py` -- `count_params`/`count_flops` (also used by
  `analysis/501_ARCADE/record_architecture_stats.py`), `count_bops` (Stage 4).
- `collect_results.py` -- after a training run: infers if needed, computes
  dice/clDice/n_components + params/FLOPs, writes/updates one `results.csv`
  row. Run manually or from the tail of a Slurm job.
- `generate_cost_tables.py` -- Stage 2.2's architecture-only marginal-cost
  tables (no training) -> `cost_tables/`.
- `plot_cost_relationships.py` -- config-independent params/FLOPs cost
  curves (per filter, per block, as a function of width) -> `cost_tables/`.
- `plot_sweep_progress.py <stage>` -- Dice vs. params/FLOPs PNG for whatever
  rows exist so far for that stage -> `sweep_progress/`.
- `smoke_test.sh` -- pre-flight smoke test (per `smoke_test_policy` in the
  yaml), run before each stage's full sweep, not a one-time gate.
- FINN/QONNX export + hardware resource estimation has moved to
  `../hardware/` (see its own README) -- `finn_resource_probe.py` (P1's
  early analytical probe), `finn_enet_prod_export.py` /
  `export_quant_checkpoint.py` (FINN-compatible model + QONNX export), and
  `finn_enet_build.py` (the working FINN estimate-only build pipeline) all
  live there now, along with the `finn_enet_deploy_xczu7ev.ipynb` notebook.
- `configs/cfinal_ops.env` -- Stage 1b's output (decoder_type + op flags),
  sourced by every downstream Slurm job so it's edited in one place.
- `slurm/` -- job arrays, one per sweep, named `<stage>_array.job`.
  `slurm/archive/` -- superseded jobs from prior plan revisions, kept for
  history (see `foundation_log.md`'s "Plan revision" section).
- `notebook/report.ipynb` -- the single reporting point; reads `results.csv`.

## Running a sweep (Stage N)

1. Make sure any upstream stage's output this one depends on is finalized
   (e.g. `configs/cfinal_ops.env` after Stage 1b).
2. Pre-flight: `bash smoke_test.sh` (or a stage-specific config passed as
   args) against the training environment.
3. Submit: `sbatch slurm/<stage>_array.job`.
4. Each task trains (skipping if its checkpoint already exists) and calls
   `collect_results.py` itself -- no separate collection pass needed.
5. Progress check any time: `python plot_sweep_progress.py <stage>`.
6. Re-run `notebook/report.ipynb` for the full per-stage tables/plots.

## Naming convention

`config_name` = `{stage}_{descriptive-suffix}`, e.g. `stage1b_e1_unpool`,
`stage2_U4_bnnative`. The nnU-Net results folder is
`nnUNetTrainerENet_{config_name}__nnUNetPlans__2d` (via `ENET_OUTPUT_FOLDER`)
-- one importable trainer class for every config, distinguished by env vars
and this folder suffix, not by trainer subclassing.
