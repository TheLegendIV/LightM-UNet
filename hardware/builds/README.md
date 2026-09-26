# hardware/builds/ — per-job build folders

Each FINN build "job" (one architecture variant, per the `AGENTS.md` naming
convention `{arch}_{decoder/op-suffix}_{warmstartNNNep}_{alphaNNN}_{dummy|trained|finn_calibrated}`)
gets its own folder here:

```
hardware/builds/<job_name>/
    finn_export_<job_name>.py              # export to QONNX
    finn_hawq_dump_conv_order_<job_name>.py # map HAWQ bit-widths onto FINN node order
    finn_hawq_preamble_<job_name>.py        # streamline + convert-to-hw + partition assignment
    finn_ooc_<job_name>_8way_full.py        # per-partition Vivado build
    finn_ooc_<job_name>_8way_per_partition_synth.py  # per-partition OOC synth
    finn_zynqbuild_<job_name>_partition0.py # optional: single-partition full PS+PL bitstream
```

Keep the full descriptive filenames (matching the top-level naming
convention already used across this repo) rather than generic
`export.py`/`preamble.py`/`build.py` — this keeps filenames self-describing
once scripts get `docker cp`'d into the FINN container's flat working
directory (see `../README.md`'s "Environment" section).

Only truly job-specific scripts belong here. Anything reused across jobs
(build-step helpers, partitioning logic, export base classes, result
collection, etc.) belongs at the top level of `hardware/` instead — see
`../README.md` for the current shared-infra list.

When a job is retired, `git mv` its folder into
`hardware/archive/<date>_<reason>/<job_name>/` rather than deleting it.

## Active jobs

- **`12_dense_relu_nearest_conv_upsample`** — the last build actually run;
  moved back here from `hardware/archive/pre_builds_refactor_20260926/`
  after the 2026-09-26 refactor archived it along with everything else.

This folder was emptied out on 2026-09-26 (see
`hardware/archive/pre_builds_refactor_20260926/` for the other 2 build
families that stayed archived: `12_dense_relu_warmstart150ep_alpha025` and
`S12_dense_nn_upsample_conv_alpha1_0`).
