# FINN container directory index

Map of the FINN installation used by everything in `hardware/`. FINN itself
lives **only inside the Docker container** (not in this git repo) — this
file exists so a future session doesn't have to re-explore it from scratch
via `docker exec`. Paths below are container-internal
(`/home/thelegendiv/finn/...`) unless noted otherwise.

Current container: check with `docker ps` — name/ID changes across
recreations (see `PARTITIONED_BUILD_LOG.md` for why). As of 2026-08-15 it's
`frosty_volhard`. FINN version: `v0.10.1-10-g39f0c9a6b-dirty` (image tag
`xilinx/finn:v0.10.1-10-g39f0c9a6b-dirty.xrt_202220.2.14.354_22.04-amd64-xrt`).

**Environment reminder**: any `docker exec <container> python3 ...` needs
`HOME=/tmp/home_dir` set, or imports of `finn`/`qonnx` fail — see
`PARTITIONED_BUILD_LOG.md`'s "Environment gotchas" section.

## Top-level layout

```
/home/thelegendiv/finn/
├── src/finn/                    # the actual finn Python package (editable install)
│   ├── builder/                 # build_dataflow.py, build_dataflow_config.py, build_dataflow_steps.py
│   ├── transformation/          # graph transforms, incl. fpgadataflow/, streamline/, qonnx/
│   ├── custom_op/fpgadataflow/  # HW custom-op defs (hls/, rtl/ backends)
│   ├── analysis/fpgadataflow/   # exp_cycles_per_layer.py, dataflow_performance.py, hls_synth_res_estimation.py
│   ├── core/                    # ModelWrapper extensions, datatype helpers
│   └── util/                    # basic.py (make_build_dir, get_vivado_root, ...)
├── deps/                        # bundled dependencies, each pip-installed --user -e
│   ├── qonnx/src/qonnx/         # ModelWrapper, custom_op registry, transformation base classes
│   ├── brevitas/                # quantization-aware training library
│   ├── finn-experimental/
│   └── pyverilator/
├── finn-rtllib/                 # hand-written Verilog for FIFO/DWC/MVU/thresholding/etc. HW ops
├── notebooks/
│   ├── enet/                    # <-- OUR WORKING DIRECTORY, see below
│   ├── basics/, advanced/, end2end_example/   # FINN's own tutorial notebooks (not ours)
├── tests/                       # FINN's own test suite (fpgadataflow/, transformation/, end2end/)
├── finn_deployment_outputs/     # leftover from an EARLIER, unrelated bitstream/estimate run
│                                 # (2026-07-27) -- not part of the current S19 partitioning work
├── tutorials/fpga_flow/
└── docker/, docs/, custom_hls/  # FINN's own container/docs infra, not used by us directly
```

## `notebooks/enet/` — our working directory

Everything we run lives here (bind-mounted from the host — survives
container recreation). Key files as of 2026-08-15:

**Build scripts (chronological lineage)**:
- `finn_enet_build.py` — original estimate-only build (no stitched IP).
- `finn_enet_ip_build.py` — first stitched-IP + OOC synth build.
- `finn_enet_build_decomposed_prelu.py` — adds the decomposed-PReLU
  streamlining steps (`step_enet_tidy`, `step_enet_streamline`, etc.) reused
  by every build script since.
- `finn_enet_ip_build_decomposed_prelu.py` — the **single-partition**
  stitched-IP build (used for S5/S13/S19 attempts, including the killed
  Attempt 6). `MODEL_NAME`/`FIFO_STRATEGY` taken from `sys.argv`.
- `finn_enet_ip_build_partitioned.py` — **current work**: the 5-way
  partitioned build (see `PARTITIONED_BUILD_LOG.md`). Imports
  `finn_stage_partition.py` + `finn_partition_build_steps.py`.
- `finn_enet_ip_resume.py` — resumes a single-partition build from a
  completed-step checkpoint via `cfg.start_step`.

**New partitioning modules** (mirrored from `hardware/` in this repo —
keep both copies in sync, repo is the source of truth):
- `finn_stage_partition.py`
- `finn_partition_build_steps.py`

**Model files** (`.onnx`): `quantEnet_s5_dscnoproj_dense_int8.onnx`,
`quantEnet_s13_leaky_frozen_int8.onnx`, `quantEnet_s19_double_mid_int8.onnx`
(current target) — exported via the corresponding
`hardware/finn_export_s*.py` scripts, then copied into the container.

**Outputs**: `finn_deployment_outputs/stitched_ip_*<config>_<fifo-strategy>_<timestamp>/`
— each build run's own directory, containing `intermediate_models/`
(per-step `.onnx` checkpoints when `save_intermediate_models=True`),
`build_dataflow.log`, and final reports/IP once done. Partitioned-build
runs use the `stitched_ip_partitioned_*` naming (set in
`finn_enet_ip_build_partitioned.py`'s `OUTPUT_DIR`).

**Diagnostic scripts** (`_tmp_diag_*`, `_tmp_check_*`, `_tmp_compile_check.py`,
`_tmp_list_onnx_nodes*.py`) — one-off investigation scripts, safe to ignore
or delete; kept only because deleting mid-investigation risked losing
useful repro steps.

## Key source files worth knowing the exact location of

- `src/finn/builder/build_dataflow.py` — `resolve_build_steps()` (steps
  are strings looked up in `build_dataflow_step_lookup`, or any `callable`
  — **must have `.__name__`**, so `functools.partial` doesn't work as a
  step; use a plain `def` wrapper instead), `build_dataflow_cfg()` main
  loop (checkpoints only between steps).
- `src/finn/builder/build_dataflow_steps.py` — generic step functions
  (`step_create_dataflow_partition` has the `assert len(sdp_nodes) == 1`
  that our partitioned flow avoids by using our own
  `step_create_dataflow_partition_multi` instead).
- `src/finn/transformation/fpgadataflow/create_dataflow_partition.py` —
  `CreateDataflowPartition`, groups nodes by `partition_id` nodeattr.
- `src/finn/transformation/fpgadataflow/set_fifo_depths.py` —
  `InsertAndSetFIFODepths`, contains the throwaway measurement
  `CreateStitchedIP` call (line ~325).
- `src/finn/transformation/fpgadataflow/create_stitched_ip.py` —
  `CreateStitchedIP`, `save_bd_design`/`save_project` only at the very end
  (line ~384).
- `src/finn/util/make_zynq_proj.py` — `MakeZYNQProject`/`ZynqBuild`, the
  proven working multi-partition loop we adapted `step_combine_partitions`
  from (interior nodes wired via direct `connect_bd_intf_net`, no DMA).
- `src/finn/custom_op/fpgadataflow/hls/iodma_hls.py` — `IODMA_hls`, inherits
  stub `get_exp_cycles`/estimation methods (all return 0/None) — DMA/DDR
  latency is a complete no-op in FINN's analytical estimates.

## Entrypoint / container internals

- `/usr/local/bin/finn_entrypoint.sh` — sets `HOME=/tmp/home_dir`, then
  `pip install --user -e` for `deps/qonnx`, `deps/finn-experimental`,
  `deps/brevitas`, `deps/pyverilator`, and `$FINN_ROOT` itself (finn proper),
  then sources Vivado/Vitis HLS `settings64.sh`, then `exec "$@"`.
- Relevant env vars (set by the entrypoint / `docker run`):
  `FINN_ROOT=/home/thelegendiv/finn`, `FINN_BUILD_DIR=/tmp/finn_dev_thelegendiv`,
  `VIVADO_PATH=/tools/Xilinx/Vivado/2022.2`, `HLS_PATH=/tools/Xilinx/Vitis_HLS/2022.2`,
  `NUM_DEFAULT_WORKERS=4` (controls FINN's internal HLS-synth worker pool
  size — relevant when deciding `max_workers` for our own
  partition-level `ProcessPoolExecutor`, to avoid oversubscribing CPU cores).
