# hardware/ — FINN hardware build pipeline

> See [`../PIPELINE.md`](../PIPELINE.md) for how this subsystem fits into the
> full pipeline (diagram, artifact interfaces, target repo layout); this file
> covers implementation detail for the hardware-build stages only.

This folder takes the Brevitas-quantized ENet-family segmentation network,
makes it FINN-compatible, exports it to QONNX, and runs it through FINN
(Xilinx/AMD's dataflow-accelerator build framework) targeting a Zynq
UltraScale+ **ZCU7EV** (`xczu7ev-ffvc1156-2-e`: 230400 LUT, 624 BRAM_18K,
1728 DSP).

**2026-09-26 refactor**: all job-specific build scripts were archived (see
`archive/pre_builds_refactor_20260926/`); the active job,
`12_dense_relu_nearest_conv_upsample` (the last one actually run), lives in
`builds/12_dense_relu_nearest_conv_upsample/`. The next new job's
export/preamble/build scripts go in `builds/<job_name>/` per
`builds/README.md`'s template; only shared infra stays at the top level.

This README replaces three older docs (`FINN_REPO_INDEX.md`,
`PARTITIONED_BUILD_LOG.md`, `resource_equivalence_int8.md`, all S19-era)
and the original single-conv "estimate-only" README — all four are kept in
`archive/` for full historical detail, but are superseded here for
day-to-day navigation.

## Folder structure

- **Top level** — only shared infra reused across every job: build-step
  helpers, partitioning logic, export base classes, result collection, and
  calibration data. See "Shared infra" below.
- **`builds/`** — one folder per job (export + HAWQ dump + preamble + build
  + OOC-synth/zynqbuild scripts for that architecture variant), named per
  the `AGENTS.md` convention. Currently holds one active job,
  `12_dense_relu_nearest_conv_upsample` — see `builds/README.md` for the
  template and naming convention for the next job.
- **`probes/`** — small, standalone stitched-IP probes (not full builds)
  used to calibrate FINN's analytical cost model against real Vivado
  synthesis on a tiny sub-network (e.g. the S12 context-block stem +
  2 bottlenecks, dense vs separable, INT4/6/8, default PE=SIMD=1 vs forced
  PE=MH/SIMD=1 folding). Each probe is a matched export+build(+OOC-synth)
  script pair.
- **`diag/`** — one-off diagnostic/investigation scripts written to chase
  down a specific bug (partition-boundary forks, weight-tensor dedup,
  imbalance-model checks, etc.). Not part of any reusable pipeline — kept
  for the record in case the same bug class resurfaces.
- **`temp/`** — ad-hoc debug scripts and throwaway intermediate artifacts
  (`_tmp_*` JSON/rpt/log/tcl dumps used to build a calibration CSV, stray
  debug stdout captures, etc.). Safe to delete/regenerate; kept only
  because some are inputs to `build_alpha025_calibration_csv.py`-style
  tools.
- **`archive/`** — retired build families: older architectures (S19,
  `8_2_relu_no_reg_w16/w20`, `26_5_w24`, `26_9_w24`, `s13`,
  `minimal_1bneck`, `decomposed_prelu`), the S12 **separable**/`min4`
  sibling variant, the parked RTL-MVAU-forcing experiment
  (`finn_enet_convert_to_hw_rtl_mvau.py` and friends — see git history /
  `finn_gotchas.md` repo memory for status), 4 historical docs
  (`FINN_REPO_INDEX.md`, `PARTITIONED_BUILD_LOG.md`,
  `resource_equivalence_int8.md`, `README_original_estimate_only_pipeline.md`),
  and — as of 2026-09-26 — 2 of the 3 most recent job families under
  `pre_builds_refactor_20260926/`: `12_dense_relu_warmstart150ep_alpha025`
  (the former "S12 dense" active build: export → hawq dump → preamble →
  8-way build → per-partition OOC synth → zynqbuild alt; see git history
  for that job's full pipeline, previously documented below in this file)
  and `S12_dense_nn_upsample_conv_alpha1_0` (the third family,
  `12_dense_relu_nearest_conv_upsample`, was moved back to `builds/` since
  it's the last build actually run).
- **`notebook/`** — Jupyter notebooks for interactive/visual inspection
  (deploy notebook, ONNX graph previews).
- **`outputs/`** — accumulated build outputs (QONNX exports, per-run
  `finn_deployment_outputs/`-style dirs copied out for safekeeping).

## Environment

FINN is **not pip-installed** — it runs from source inside its own Docker
container (separate from the training container and from Vivado/Vitis).
The container ID/name changes across recreations — check `docker ps`
(recent name: `practical_davinci`). Every script that touches FINN/QONNX
inserts these to `sys.path` before importing:

```
/home/thelegendiv/finn/src
/home/thelegendiv/finn/deps/qonnx/src
/home/thelegendiv/finn/deps/brevitas/src
/home/thelegendiv/finn/deps/pyverilator
/home/thelegendiv/finn/deps/finn-experimental
```

`docker exec` needs `-e HOME=/tmp/home_dir` or `finn`/`qonnx` imports fail.
Vivado/Vitis HLS 2022.2 live at `/tools/Xilinx/{Vivado,Vitis_HLS}/2022.2`
inside the container but are **not** on `PATH` by default — `source
/tools/Xilinx/Vivado/2022.2/settings64.sh` first. All scripts are deployed
by `docker cp`-ing the individual `.py` file into the container's flat
`/home/thelegendiv/finn/notebooks/enet/` working directory (this repo's
local folder layout, incl. the `probes/`/`diag/`/`temp/`/`archive/` split,
is purely a local git-organization concern and does not need to mirror the
container's flat layout).

See repo memory `finn_gotchas.md` / `finn-container-env.md` for the much
longer list of accumulated gotchas (DSP-field JSON parser bug, fork-boundary
partitioning bugs, container-loss recovery, etc.) — this README only covers
what's needed to navigate the folder.

## S12 dense pipeline (what's needed to (re)generate a build)

1. **Export to QONNX** — `finn_export_12_dense_relu_warmstart150ep_alpha025_dummy.py`
   (fresh/untrained weights, fast iteration) or `..._trained.py` (real
   checkpoint weights) or `..._finn_calibrated.py` (calibration variant).
   Uses `finn_enet_prod_export.py` (shared `FINNQuantENet` base class) and
   `export_quant_checkpoint.py` (loads a trained checkpoint's weights).
2. **Dump the HAWQ conv order** — `finn_hawq_dump_conv_order_12_dense_relu_warmstart150ep_alpha025.py`
   (maps HAWQ per-layer bit-width search results onto FINN's node order).
3. **Run the preamble** (streamline + convert-to-hw + 8-way partition
   assignment, no Vivado) — `finn_hawq_preamble_12_dense_relu_warmstart150ep_alpha025_trained.py`
   (`_ftmainup`/`_finn_calibrated`/`_dummy_initialblock_check` are
   variant/debug preambles for the same build family). Uses the shared
   `finn_enet_build.py` (custom tidy/streamline/convert-to-hw steps),
   `finn_stage_partition.py` (partition-boundary computation), and
   `finn_enet_ip_build_partitioned_8way.py` (8-way build-step list).
4. **Run the real 8-way Vivado build** —
   `finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_8way_full.py`
   (per-partition `step_specialize_layers` → folding → codegen → IP-gen →
   `CreateStitchedIP`, parallelized via `ProcessPoolExecutor`; includes the
   HAWQ-folding-config bridge from logical layer names to FINN node names).
   Uses `finn_partition_build_steps.py` for the shared per-partition step
   functions.
5. **OOC-synthesize each partition** —
   `finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_partition_synth.py`
   (real Vivado `SynthOutOfContext` + rtlsim per partition, no combined
   bitstream — see `finn_gotchas.md` for why per-partition beats combining).
6. **Alternative: single-partition ZynqBuild** —
   `finn_zynqbuild_12_dense_relu_alpha025_partition0.py` /
   `launch_zynqbuild_partition0.sh` for a full PS+PL bitstream of one
   partition (board bring-up / driver testing, not the 8-way split).
7. **Fine-tuning support** —
   `finetune_main_up_12_dense_relu_warmstart150ep_alpha025.py` /
   `testbench_bilinear_vs_nearest_depthwise_up4up5.py` /
   `verify_bilinear_kernel.py` — the frozen nearest+depthwise substitute for
   `main_up`'s bilinear ConvTranspose, and its numerical verification.

**Shared infra** (used by the pipeline above, keep even though not
dense-specific by name): `finn_enet_build.py`, `finn_stage_partition.py`,
`finn_partition_build_steps.py`, `finn_enet_ip_build_partitioned_8way.py`,
`finn_enet_prod_export.py`, `export_quant_checkpoint.py`,
`run_export_in_container.py`, `dump_node_attrs.py`, `collect_results.py`.

**Calibration data**: `results.csv` (per-build resource/perf summary, see
`collect_results.py`), `mvau_lut_calibration_dataset.csv` (per-node
LUT/DSP/BRAM calibration data). `mvau_lut_correlation_report.txt` is a
derived correlation summary.

## Probes (`probes/`)

Matched export+build(+OOC-synth) script triples for tiny, fast-iterating
stitched-IP-only probes, used to calibrate FINN's analytical cost model
against real Vivado numbers without paying for a full 8-way build:

- `finn_export_probe_s12_context_{dense,separable}_int{4,6,8}.py` +
  `finn_export_probe_s12_context_common.py` — export the S12 context-block
  stem + 2 bottlenecks at a given precision.
- `finn_build_probe_s12_context_int4.py` (original INT4-only script) and
  `finn_build_probe_s12_context_v2.py` (generalized: `--bitwidth {6,8}` +
  `--force-pe-mh-simd1` to force PE=MH/SIMD=1 folding instead of the
  default PE=SIMD=1) + `finn_ooc_probe_s12_context_synth.py`.
- `finn_export_probe_upsample_nearest_depthwise_int8.py` +
  `finn_build_probe_upsample_nearest_depthwise_int8.py` +
  `launch_probe_upsample_nearest_depthwise.sh` — isolated probe that found
  and fixed the mid-graph `InferUpsample`/NHWC-layout bug (see
  `finn_gotchas.md`).
- `build_probe_calibration_csv.py` — builds a per-MVAU calibration CSV
  (same schema as the top-level `mvau_lut_calibration_dataset*.csv`) from a
  probe's Vivado hierarchical utilization report + landed ONNX nodeattrs
  (no folding-config JSON needed, since probes force PE/SIMD directly).

## Everything else

See `archive/` for retired build families (S19, separable/min4 sibling
builds, older architectures, the parked RTL-MVAU experiment) and the 4
historical docs moved there. See `diag/` and `temp/` for one-off
investigation/debug scripts kept only for reference.


