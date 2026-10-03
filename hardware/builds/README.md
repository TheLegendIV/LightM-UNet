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

- **`12_dense_relu_nearest_conv_upsample_256`** — v1, the last build actually
  run; moved back here from `hardware/archive/pre_builds_refactor_20260926/`
  after the 2026-09-26 refactor archived it along with everything else.
- **`12_dense_relu_nearest_conv_upsample_256_v2`** — same architecture,
  v2 MILP solve (`--optimize-downstream-rate 1.5`), checkpoint fine-tuned
  DIRECTLY on `LayerQuantEnetFINN` (not converted post-hoc like v1). Own
  job folder per this README's convention rather than sharing v1's; each
  job's export/conv_order.json artifacts live under that job's own
  `outputs/` subfolder (NOT the shared/deprecated
  `hardware/outputs/finn_exports/` — that top-level dir is being phased
  out in favor of per-job `outputs/`).
- **`12_dense_relu_nearest_conv_upsample_256_w8_16_v2`** — w8/16 width
  point of the same architecture (`CHANNELS=(4, 8, 16, 8, 4)` instead of
  `(4, 16, 32, 16, 4)`; see `MILP/configs/config_12_dense_relu_nearest_
  conv_upsample_256_w8_16.py`), checkpoint
  `nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_conv_upsample_256_w8_16_perlayer_..._w8_16_v3`
  (also fine-tuned DIRECTLY on `LayerQuantEnetFINN`), MILP solve under
  `MILP/artifacts/S12_dense_nn_upsample_256_w8_16_v2/`
  (`--optimize-downstream-rate 2.25`, matching what the `_v3` QAT job
  actually trained against). Every pipeline-stage script is otherwise
  byte-for-byte structurally identical to `..._256_v2`'s (architecture
  shape/partitioning/DSP-forcing/URAM-budget logic is channel-width
  agnostic) — only `CHANNELS`, checkpoint/bits-file paths, and the dedicated
  `finn_build_tmp/S12_dense_nn_upsample_256_w8_16_v2/` build dir differ.
- **`12_dense_relu_nearest_conv_upsample_256_w8_16_v4`** — w8/16 width point,
  its OWN dedicated QAT checkpoint (NOT a reuse of `..._w8_16_v2`'s export —
  an earlier "pure new solve" assumption was corrected), fine-tuned via
  `compression/slurm/qat_12_dense_relu_nearest_conv_upsample_256_w8_16_v4.job`
  against the v4 MILP solve (`MILP/artifacts/S12_dense_nn_upsample_256_w8_16_v4/`,
  `--max-bram-fraction 0.2 --target-fps 200 --optimize-downstream-rate 2.5`).
  Has its own `finn_export_..._w8_16_v4_trained.py`,
  `finn_hawq_dump_conv_order_..._w8_16_v4.py`,
  `finn_hawq_preamble_..._w8_16_v4_trained_256x256.py`, and
  `finn_ooc_..._v4_256x256.py` (dedicated `finn_build_tmp/.../w8_16_v4/`
  build dir) — no artifacts shared with `..._w8_16_v2`.
This folder was emptied out on 2026-09-26 (see
`hardware/archive/pre_builds_refactor_20260926/` for the other 2 build
families that stayed archived: `12_dense_relu_warmstart150ep_alpha025` and
`S12_dense_nn_upsample_conv_alpha1_0`).
