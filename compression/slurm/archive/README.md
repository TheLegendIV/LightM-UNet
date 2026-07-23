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
