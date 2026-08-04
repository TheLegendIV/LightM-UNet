# Archived compression/ scripts

Kept for history, not part of the active plan (same convention as
`compression/slurm/archive/`).

- `generate_hardware_savings_ranking.py` -- analytical/placeholder ranking
  over the binary run's section-2a grid specifically (hardcoded
  `Original/O2/O4/O8/O16/OF x native/5/3/2` filter/bottleneck axis on
  `Dataset501_ARCADE`). Already superseded during the binary run itself by
  `rank_results.py` (see that script's own docstring), which ranks whatever
  is actually in `results.csv` generically instead of a hardcoded grid --
  and now also carries this session's params/MACs/Dice-only scoring for the
  4-class objective. Not adapted to the new objective because there's
  nothing left for it to do that `rank_results.py` doesn't already do
  better; its own hardcoded grid doesn't correspond to anything in the new
  stage_1/2/3/4 structure anyway.
- `generate_symmetric_reduction_family.py` -- another binary-run/section-2a-
  specific analytical script (f_i-fixed reduction family off "Original",
  0.5/0.5 macs/activation-memory score, no Dice term at all). Same
  supersession story as `generate_hardware_savings_ranking.py`, plus it
  still uses the activation-memory axis this session dropped from
  `rank_results.py`'s formula entirely -- not adapted for the same reasons.
  Its stale outputs (`cost_tables/{activation_memory,hardware_savings_ranking,
  symmetric_reduction_family}.{csv,png}`) were removed alongside it -- none
  of the three still-active scripts (`generate_cost_tables.py`,
  `plot_cost_relationships.py`, `rank_results.py`) produce them anymore.
