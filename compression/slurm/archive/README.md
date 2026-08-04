# Archived Slurm jobs

Superseded by the 37-run structured plan (see `compression/foundation_log.md`'s
"Plan revision" section) -- kept for history, not part of the active plan.

- `stage1b_structure_array.job` -> replaced by `pre_pruning_1b_maxunpool_array.job`
  + `pre_pruning_1c_specialop_array.job`
- `stage2_grid_array.job` -> replaced by `pruning_2a_grid_array.job`
- `stage2_4_fi_reduction_array.job` -> replaced by `pruning_2b_fi_tune_array.job`
- `early_probe_p2_array.job` -> replaced by `postpruning_3a_quant_array.job`
  (no direct equivalent of a fully decoupled early probe in the new plan --
  quantization now happens only after pruning, as section 3a)

## Binary Dataset501 sweep + baseline jobs -- superseded by the Dataset509 4-class pivot

The compression experiment's objective changed from binary (single
foreground class) coronary-vessel segmentation on `Dataset501_ARCADE` to
4-class segmentation (LAD/RCA/LCX/LM) on `Dataset509_ARCADE_1x1_4c` (see
`compression/README.md`'s top note and `agent_instructions_1.yaml`'s top
note). None of these jobs has a direct 1:1 replacement in the old stage
numbering -- the new objective restarts its own stage sequence from
`stage_1_naive_baseline_array.job` onward (see `compression/README.md`).
Kept for history (real Dice/params/FLOPs numbers on the binary task that
informed the findings in `foundation_log.md`), not part of the active plan.

- `pre_pruning_1a_seed_array.job`
- `pre_pruning_1b_maxunpool_array.job`
- `pre_pruning_1c_dilated_only.job`
- `pre_pruning_1c_specialop_array.job`
- `pre_pruning_1d_relu_only.job`
- `pruning_2a_grid_array.job`
- `pruning_2b_fi_tune_array.job`
- `postpruning_3a_quant_array.job`
- `postpruning_3a_quant_O4_int8_no_asym.job`
- `resume_3a_quant_O4_int8_no_asym.job`
- `train_enet_e1.job` (originally at repo root -- moved here since this is
  the only archive location in the repo and it shares the same
  binary-Dataset501 fate as the rest of this list)
- `train_enet_original.job` (repo root, same as above)
