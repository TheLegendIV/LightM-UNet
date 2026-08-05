# hardware/ — FINN hardware-estimation pipeline for QuantENet

This folder holds everything needed to take the Brevitas-quantized ENet
(`enet/nnunetv2/nets/QuantENet.py`), make it FINN-compatible, export it to
QONNX, and run it through FINN's *analytical* resource/performance estimator
targeting a Xilinx **ZCU7EV** (`xczu7ev-ffvc1156-2-e`: 48000 LUT, 216 BRAM,
192 DSP). No Vivado/Vitis synthesis or bitstream is produced — this is the
"does the architecture roughly fit, and how fast would it run" step
(`agent_instructions_1.yaml`'s `early_probes.p1_finn_resource_probe` /
`stage_5_finn_folding`'s analytical-estimate slice).

This folder was split out from `compression/` because it's a separate
concern (hardware feasibility) from the software compression sweep
(pruning/quantization for accuracy vs. cost), which stays in
`compression/`. A handful of one-off debugging scripts used only to
diagnose the bug chain described below have been deleted now that the
pipeline works — see "Issues encountered" for what they found, kept here
for the record.

## File map

- `finn_enet_prod_export.py` — defines `FINNQuantENet`, a FINN-compatible
  reimplementation of `QuantENet.py`, and exports it to QONNX.
- `export_quant_checkpoint.py` — exports an actual **trained** checkpoint's
  weights (not just a fresh random init) to QONNX, looking up its
  architecture from `compression/results.csv` by `config_name`.
- `run_export_in_container.py` — driver script that runs
  `finn_enet_prod_export.py` inside the FINN container and writes the
  resulting `.onnx` into the container's `finn/notebooks/enet/` folder
  (where `finn_enet_build.py` expects to find it).
- `finn_enet_build.py` — **the working deliverable**: a standalone
  estimate-only FINN build script with a custom streamlining/HW-conversion
  pipeline tuned for ENet's residual connections and `ConvTranspose`
  upsampling. Run this inside the FINN container; it prints performance and
  resource estimates at the end.
- `finn_resource_probe.py` — an earlier, simpler probe (P1) that exports a
  vanilla `QuantENet.py` model (not the FINN-specific variant below) and
  was used to first verify the QONNX-export half of the pipeline, before
  the FINN Docker container was available. Superseded by
  `finn_enet_build.py` for anything beyond a quick export sanity check —
  `QuantENet.py`'s un-modified topology (asymmetric convs, `MaxUnpool`,
  etc.) is **not** guaranteed to make it through FINN's dataflow
  conversion; only the `FINNQuantENet` variant has been proven to.
- `notebook/finn_enet_deploy_xczu7ev.ipynb` — interactive version of the
  same estimate-only build, useful for stepping through intermediate models
  cell-by-cell.
- `outputs/qonnx_exports/` — QONNX exports of trained checkpoints (from
  `export_quant_checkpoint.py`).
- `outputs/early_probes/` — QONNX export from the early `finn_resource_probe.py`
  probe.

## Environment

FINN is **not pip-installed** — it's run from source inside its own Docker
container (separate from the training container and from Vivado/Vitis),
container id referenced in these scripts as `d345f89b4e6c`. Every script
that touches FINN/QONNX inserts these to `sys.path` before importing:

```
/home/thelegendiv/finn/src
/home/thelegendiv/finn/deps/qonnx/src
/home/thelegendiv/finn/deps/brevitas/src
/home/thelegendiv/finn/deps/pyverilator
/home/thelegendiv/finn/deps/finn-experimental
```

## Workflow (steps taken)

1. **Export a FINN-compatible ENet to QONNX** — `finn_enet_prod_export.py`,
   run inside the container via `run_export_in_container.py`. Produces
   `quantEnet_finn_v1.onnx` (channels `(20, 72, 144, 72, 20)`, bottlenecks
   `(4, 8, 8, 2, 1)` — see "Which config was exported" below).
2. **Run the FINN estimate-only build** — `finn_enet_build.py`, which pushes
   the QONNX model through `step_qonnx_to_finn` → custom tidy/streamline/
   convert-to-hw steps → `step_create_dataflow_partition` →
   `step_specialize_layers` → `step_target_fps_parallelization` →
   `step_apply_folding_config` → `step_minimize_bit_width` →
   `step_generate_estimate_reports`.
3. **Debug the conversion pipeline** (see below) until the whole chain ran
   without errors and produced complete JSON reports.
4. **Read off the estimate reports** and compare totals against the ZCU7EV
   budget (printed directly by `finn_enet_build.py`).

Standard FINN builds (`finn.builder.build_dataflow_config`) get almost this
whole pipeline for free via `step_tidy_up` / `step_streamline` /
`step_convert_to_hw`. Those generic steps don't handle this network's
combination of residual `Add`s and `ConvTranspose` upsampling correctly, so
`finn_enet_build.py` replaces them with custom equivalents (see next
section) rather than trying to patch the generic ones in place.

## ENet(Quant) version this is based on

- **Base topology**: `enet/nnunetv2/nets/ENet.py`, the standard ENet
  encoder/decoder (initial block, downsampling/regular/upsampling
  bottlenecks, dilated context stages), as already used and validated
  (self-tested, trained) elsewhere in this repo.
- **Quantization**: `enet/nnunetv2/nets/QuantENet.py` mirrors `ENet.py`
  block-for-block, adding Brevitas `QuantConv2d`/`QuantConvTranspose2d`
  (INT8 weights, `Int8WeightPerTensorFloat`) and quantized activations,
  homogeneous bit-width only (no per-layer mixed precision yet). One
  deliberate deviation already baked in at that layer: **`PReLU` →
  `QuantReLU`**, because Brevitas/FINN has no standard quantized `PReLU`
  op — so Dice numbers from `QuantENet` are not directly comparable to the
  FP32 `ENet` baseline on activation function alone, independent of
  quantization itself. See "PReLU vs. QuantReLU — FINN compatibility
  investigation" below: direct `PReLU` is confirmed permanently
  incompatible, but a drop-in, FINN-native, numerically-equivalent
  decomposition has been found and proven (not yet integrated here).
- **FINN-specific variant**: `finn_enet_prod_export.py`'s `FINNQuantENet`
  takes `QuantENet`'s topology one step further, specifically to survive
  FINN's dataflow/streamlining transforms (see next section for exactly
  what changed and why).
- **Config actually exported/built**: `channels=(20, 72, 144, 72, 20)`,
  `bottlenecks_per_stage=(4, 8, 8, 2, 1)`, `bit_width=8` — this is the
  **`E1` cell** from `agent_instructions_1.yaml`'s `stage_2_architecture_grid`
  filter axis (the largest/baseline candidate, not a pruned/compressed
  winner). It was chosen to first prove the FINN pipeline works at all on
  a real-size network; it is **not** `best_model.env`'s pruning-selected
  config (`channels=(16, 8, 16, 8, 4)`). See "Uncertainty" below — this is
  the main reason the resource estimate is so far over budget.
- **Weights**: the QONNX file the successful build/estimate ran against was
  exported from a **freshly initialized (untrained), fixed-seed
  (`torch.manual_seed(0)`) model**, not a trained checkpoint. This is fine
  for FINN's *resource* estimate (LUT/BRAM/DSP/cycles depend on
  architecture + bit-width, not on weight values), but it means **no
  accuracy number exists yet for this exact FINN-compatible variant** —
  `export_quant_checkpoint.py` exists specifically to re-export a *trained*
  checkpoint's weights once one exists for a FINN-compatible architecture.

## What had to change in software to satisfy FINN

These are real topology deviations from `QuantENet.py`, not just export
quirks — each one is documented in `finn_enet_prod_export.py`'s module/class
docstrings too:

1. **No asymmetric convolutions.** FINN's `ConvolutionInputGenerator` was
   not trusted to handle asymmetric kernels cleanly in every configuration,
   so `FINNQuantENet` only uses square kernels.
2. **No `MaxUnpool`, no `F.interpolate`-based upsampling.** `MaxUnpool2d`
   isn't exportable to ONNX at all (`torch.onnx.export` raises
   `UnsupportedOperatorError` for `aten::max_unpool2d`); bilinear
   `F.interpolate` becomes an ONNX `Resize` node, which FINN's dataflow
   backend doesn't support. **Fix**: the upsampling bottleneck's main path
   uses `QuantConvTranspose2d(kernel=2, stride=2)` instead — a learned
   upsample that also does the channel projection in one op.
3. **`MaxPool` shortcut without `return_indices`.** The original
   `QuantDownsamplingBottleneck` always calls `MaxPool(return_indices=True)`
   even when the decoder doesn't need indices (`upsample_conv` mode); the
   unused `indices` output left a dangling ONNX graph output FINN didn't
   handle well. **Fix**: shortcut path uses
   `MaxPool(return_indices=False)` + a 1×1 `QuantConv` for the channel
   projection instead of reusing pooled indices.
4. **Initial block: single conv, no `MaxPool`+`Concat`.** The original ENet
   `InitialBlock` concatenates a stride-2 conv branch with a `MaxPool`
   branch. `MaxPool`→`Concat` isn't cleanly handled by FINN's dataflow
   partitioner, so `FINNInitialBlock` uses one stride-2 3×3 conv that
   directly produces the full output channel count instead.
5. **Final layer has no bias.** A biased final layer complicates
   streamlining (bias folding into thresholds needs the output to be
   quantized, which the segmentation head's output isn't). The final
   `QuantConvTranspose2d` is `bias=False`.
6. **No depthwise-separable convolutions (DSC).** Not tested against FINN
   for this model variant at all — left out rather than risk an unverified
   op.
7. **Dilated convolutions were kept** (`use_dilated=True` is fine) — FINN's
   `ConvolutionInputGenerator` supports a `dilations` parameter natively, so
   no change was needed there.

None of these are yet re-validated for Dice/accuracy — they're topology
changes made purely to satisfy FINN's ONNX/dataflow constraints. If this
FINN-compatible variant is trained, its accuracy needs to be re-measured
independently of the software-side compression sweep's numbers (which use
`QuantENet.py`/`ENet.py`, not `FINNQuantENet`).

## Custom FINN build pipeline: decisions & custom work

`finn_enet_build.py` does **not** use FINN's generic
`step_tidy_up`/`step_streamline`/`step_convert_to_hw`. It defines its own
`step_enet_tidy` / `step_enet_streamline` / `step_enet_convert_to_hw`,
modelled on `finn-examples/build/resnet50/custom_steps.py` since ResNet50
is the FINN reference example that also has to handle residual `Add`s:

- **4 alternating iterations** of a "linear" pass (`MoveAddPastMul`,
  `MoveScalarMulPastMatMul/Conv/ConvTranspose`, `MoveMulPastMaxPool`,
  threshold absorption, etc.) and a "non-linear" pass
  (`MoveLinearPastFork` → `MoveLinearPastEltwiseAdd`) — repeated because a
  single pass doesn't fully push every scale/bias past every fork+Add in
  one go across a network this deep.
- **`Conv`→`MatMul` lowering (`LowerConvsToMatMul`) happens inside the
  custom streamline step**, not the generic convert-to-hw step, so that
  `InferDataTypes` can correctly type `Im2Col` outputs as integer *before*
  `InferQuantizedMatrixVectorActivation` needs them. Left in the standard
  location, all lowered convolutions would default to `FLOAT32` and never
  convert to `MVAU`.
- **`ConvTranspose` → HW conversion via `InferPixelPaddingDeconv`** (see bug
  #3 below) — this was a deliberate, non-obvious addition; FINN has no
  direct HW layer for `ConvTranspose`, and this transform was found by
  searching FINN's source tree for anything that mentions deconvolution,
  not from any tutorial or documented pattern.
- **The build step list explicitly matches** FINN's canonical
  `estimate_only_dataflow_steps` (`finn/src/finn/builder/build_dataflow_config.py`),
  including `step_specialize_layers` and `step_minimize_bit_width` — both of
  which are easy to forget when hand-rolling a step list, and both of which
  cause silent/confusing failures later in the pipeline rather than an
  obvious error at the point they're missing (see bug #4 below).
- **`mvau_wwidth_max=80`, `target_fps=10000`, `synth_clk_period_ns=10.0`**
  are the current build config values — `target_fps=10000` in particular
  was **not** chosen from any real throughput requirement, it's a
  placeholder that FINN's `step_target_fps_parallelization` used to decide
  how aggressively to parallelize (fold) each layer. This is the direct
  cause of the current resource overage (see "Uncertainty").

## Issues encountered & fixes

The build originally failed with `AssertionError: cycle-free graph
violated: partition depends on itself` in `step_create_dataflow_partition`.
Root-causing this took bisecting the streamlining pipeline transform-by-
transform against intermediate `.onnx` files (FINN saves one per step when
`save_intermediate_models=True`). Six distinct issues were found and fixed,
in the order they were uncovered:

1. **`MoveLinearPastEltwiseAdd` running before `MoveLinearPastFork`.**
   `MoveLinearPastEltwiseAdd.move_node()` renames a `Mul`/`Add` producer's
   output tensor without checking whether that producer also feeds another
   consumer (a fork point — e.g. a shortcut-branch `Conv`). Run first, it
   silently corrupts the other consumer's input reference into a dangling
   tensor name with no producer, which only manifests much later as
   `FLOAT32`-typed `Conv`/`MatMul` inputs blocking `MVAU` conversion. **Fix:**
   always run `MoveLinearPastFork()` before `MoveLinearPastEltwiseAdd()`
   (`_streamline_nonlinear`).
2. **`AbsorbTransposeIntoMultiThreshold()` right after
   `LowerConvsToMatMul()`.** This removes one half of a back-to-back
   `Transpose` pair, orphaning the other's input. **Fix:** removed; let
   `AbsorbConsecutiveTransposes()` cancel the pair directly instead.
3. **`ConvTranspose` has no native FINN HW layer.** Left as a generic ONNX
   op interleaved in the *middle* of the network (not just at the
   input/output boundary), it created disjoint non-HW islands that broke
   `step_create_dataflow_partition`'s single-partition assumption — this
   was the direct cause of the "cycle-free graph violated" error. **Fix:**
   `InferPixelPaddingDeconv` (found in
   `finn/src/finn/transformation/fpgadataflow/infer_pixel_padding_deconv.py`)
   rewrites `ConvTranspose` (group=1 only — confirmed true for every
   `ConvTranspose` in this network) into
   `Transpose`+`FMPadding_Pixel`+`Im2Col`+`MatMul`+`Transpose`, the same
   pattern `LowerConvsToMatMul` uses for regular `Conv`, so it converts to
   `ConvolutionInputGenerator`/`MVAU` through the normal path afterward.
4. **Missing `step_specialize_layers` / `step_minimize_bit_width` in the
   custom build step list.** Without `step_specialize_layers`, HW nodes
   stay generic (not HLS/RTL-backed), and FINN's `dataflow_performance`
   analysis silently returns an empty dict, surfacing as
   `ValueError: max() arg is an empty sequence` deep inside
   `step_target_fps_parallelization` — a confusing error far from its
   actual cause. **Fix:** step list now matches FINN's own
   `estimate_only_dataflow_steps` exactly.
5. **A `Mul` (dequant scale) stuck directly before `MaxPool` or
   `ConvTranspose`** never got absorbed, because
   `AbsorbMulIntoMultiThreshold` only handles `Mul`→`MultiThreshold`. **Fix:**
   added `MoveMulPastMaxPool()` and `MoveScalarMulPastConvTranspose()` to
   the linear streamlining pass so these keep moving forward until they
   reach something that can absorb them.
6. **`GiveReadableTensorNames()` called mid-loop** during the 4-iteration
   streamlining, which risked renaming a tensor without updating every
   consumer reference and creating more phantom tensors. **Fix:** only
   called once, after all streamlining is complete.

## Current results

Estimate-only build succeeds end-to-end and produces full reports
(`report/estimate_network_performance.json`,
`estimate_layer_resources.json`, `estimate_layer_cycles.json`,
`estimate_layer_config_alternatives.json`, `op_and_param_counts.json`) —
328 total nodes, 325 converted to HW ops (`AddStreams`,
`ConvolutionInputGenerator`, `DuplicateStreams`, `FMPadding`,
`FMPadding_Pixel`, `MVAU`, `StreamingMaxPool`, `Thresholding`), 3 left as
non-HW (input-layout `Transpose`, output-layout `Transpose`, and the final
dequantization `Mul` — all expected/benign, outside the dataflow
partition).

At the `E1` config and default (unfolded) parallelism:

| Metric | Value | ZCU7EV budget | % of budget |
|---|---|---|---|
| LUT | 1,033,620 | 48,000 | 2153% |
| DSP | 2,390 | 192 | 1245% |
| BRAM | 0 | 216 | 0% |
| Estimated throughput | ~3,052 fps | — | — |
| Estimated latency | ~44.8 ms | — | — |
| Critical path | 4,482,964 cycles | — | — |
| Bottleneck node | `MVAU_rtl_80` (32,768 cycles) | — | — |

## PReLU vs. QuantReLU — FINN compatibility investigation (2026-08-05)

Investigated whether ENet's real activation (`nn.PReLU`, per-channel) can be
made FINN-hardware-compatible, instead of the `QuantReLU` substitution
`QuantENet.py` currently hardcodes everywhere. Used a minimal test harness
(`hardware/_tmp_prelu_investigation2.py`, kept for reference) that reuses
ENet's actual `QuantInitialBlock`/`QuantRegularBottleneck` code verbatim
with a swappable activation factory, so PReLU is the only variable against
a proven-working `QuantReLU` control.

### Result: direct `PReLU` is permanently incompatible; a decomposition works

1. **Direct/naive `PReLU` is a hard, categorical block — confirmed two ways:**
   - Source grep across the entire FINN + QONNX codebase finds zero
     references to `PRelu` anywhere. FINN's complete `fpgadataflow` HW op
     registry (`finn/src/finn/custom_op/fpgadataflow/__init__.py`) has
     exactly 20 registered types (`MVAU`, `Thresholding`, `VVAU`,
     `AddStreams`, `ChannelwiseOp`, `StreamingEltwise`, etc.) — none can
     represent `PRelu`.
   - Empirically: a literal ONNX `PRelu` node survives `step_qonnx_to_finn`
     and the *entire* streamlining + HW-conversion pipeline completely
     untouched — it is never absorbed, converted, or rewritten by anything,
     and remains a permanently-stuck non-HW node.
2. **A decomposition into ops FINN already supports does work.** The
   naive textbook rewrite `ReLU(x) - alpha*ReLU(-x)` doesn't help — negation
   always traces to ONNX `Neg` (whether written as `-x` or `x * (-1.0)`),
   and `Neg` is equally unsupported (also zero grep hits). Instead, the
   **algebraic identity**
   `PReLU(x) = alpha_c * x + (1 - alpha_c) * ReLU(x)`
   (exact, not an approximation — trivially verified by cases: `x>=0` gives
   `x`, `x<0` gives `alpha_c*x`) uses only: one quantized ReLU (native,
   `MultiThreshold`→`Thresholding`), a fork of `x` into two branches, two
   per-channel `Mul` scales (→ `InferChannelwiseLinearLayer`), and one
   `AddStreams` combine of the two branches — the same fork+recombine shape
   ENet's own residual connections already use successfully.
3. **Two real implementation gotchas found along the way:**
   - **Mul operand order matters for FINN's constant-folding transforms.**
     Writing `alpha * x` (constant first) traces the constant as ONNX
     input[0], but `CollapseRepeatedMul` and friends hard-assume the
     constant is always input[1] — trips
     `AssertionError: Initializer for parameters for op1 is not set`. Always
     write `x * alpha` (dynamic first).
   - A separate, **PReLU-unrelated generic bug** blocked all variants
     (including the plain-ReLU control!) at `step_enet_convert_to_hw`:
     `AssertionError: MultiThreshold_0: Signed output requires actval < 0`.
     Root cause: a degenerate/uncalibrated `out_bias` on the input
     quantizer's threshold node in this tiny, untrained 2-block test
     network — not a FINN limitation (the real, full-depth network already
     completed a genuine 66-hour hardware build with the same op
     vocabulary). Patched for testing purposes by forcing
     `out_bias = -(2**(bits-1))` on any signed `MultiThreshold` with a
     non-negative bias.
4. **After both fixes, all three variants (control / naive PReLU /
   decomposed PReLU) were pushed through the same `step_enet_convert_to_hw`.**
   Control and the decomposition both leave behind only ordinary,
   already-elsewhere-converted node types as residue (`Transpose`, `Mul`,
   `Add`, `MaxPool`, `Im2Col`, `MatMul`, `Concat`, `MultiThreshold` — the
   kind of thing later production steps mop up, not included in this
   reduced test step list). The naive-PReLU variant's residue explicitly
   still contains the literal, permanently-stuck `PRelu` node — the one
   categorical difference, proving the decomposition reaches full
   op-vocabulary parity with the working baseline.
5. **Numeric correctness check (separate from the graph-conversion check).**
   Comparing the actual quantized decomposition module's forward pass
   against real `nn.PReLU` on properly-scaled random input gave error
   statistically identical to naive-requantizing real PReLU's own output
   (max abs err 0.238, mean abs err ~0.002 on a ~1.54 output range —
   ordinary 8-bit quantization noise, nothing decomposition-specific).

### What this does *not* yet prove

- No full hardware build was attempted (`step_create_dataflow_partition`
  onward, real HLS/RTL codegen, Vivado synthesis) — only a reduced
  `step_enet_convert_to_hw` pass on a 2-block slice, not the complete
  production recipe on the full-depth network.
- The decomposition has not been wired into the real `QuantENet.py` /
  `finn_enet_ip_build.py` production pipeline.
- No trained-checkpoint accuracy comparison (Dice/clDice) between real
  PReLU and this decomposition has been run.

## Uncertainty / open items

- **This resource estimate is expected to fail budget** — `E1` is the
  largest candidate in the compression sweep's filter axis, not the
  pruned/selected winner (`best_model.env`), and `target_fps=10000` was a
  placeholder that pushed `step_target_fps_parallelization` toward maximum
  parallelism (PE/SIMD), not a real target. The huge LUT/DSP overage
  (21.5×/12.4×) is not evidence the architecture can't fit — it hasn't been
  folded down yet. **Next step:** either lower `target_fps` substantially
  or supply an explicit `folding_config_file` with conservative per-layer
  PE/SIMD, then re-run and check against budget.
- **BRAM usage is 0%** — worth double-checking once folding is reduced;
  either the estimator is putting all weight storage in LUTRAM/distributed
  RAM at this config, or something about threshold/weight storage isn't
  being counted the way it would be for a folded, BRAM-backed design.
- **No accuracy validation yet for `FINNQuantENet`.** The QONNX file this
  was estimated against has untrained (random, seeded) weights — resource/
  cycle counts are weight-independent so this doesn't affect the numbers
  above, but Dice/clDice for this exact FINN-compatible topology (with the
  7 changes listed above vs. `QuantENet.py`) has not been measured. It
  should not be assumed to match the software-sweep's `QuantENet.py`
  numbers.
- **`fpga_part="xczu7ev-ffvc1156-2-e"`** is a common default part string for
  "ZCU7EV" but has not been individually confirmed against the exact
  board/package/speed-grade this project targets.
- **This is estimate-only** — no Vivado/Vitis synthesis has been run, so
  these are FINN's analytical model's numbers, not post-synthesis/post-P&R
  numbers. Real LUT/DSP/Fmax after synthesis can differ (usually somewhat
  higher for LUTs due to routing/control logic FINN's analytical model
  doesn't fully capture).
- **`finn_resource_probe.py` is now a secondary/legacy tool** — kept for
  reference and quick sanity re-checks of the QONNX-export path, but it
  exports plain `QuantENet.py` (which still has asymmetric convs,
  `MaxUnpool`, etc.) rather than the FINN-safe `FINNQuantENet`, so its
  output should not be pushed through the full `finn_enet_build.py`
  pipeline without expecting it to hit the same issues documented above.

## Re-running

```bash
# 1. Export a fresh QONNX model into the container (edit channels/bnecks in
#    run_export_in_container.py first if needed):
docker cp hardware/finn_enet_prod_export.py <container>:/tmp/
docker cp hardware/run_export_in_container.py <container>:/tmp/
docker exec <container> python /tmp/run_export_in_container.py

# 2. Run the estimate-only build:
docker cp hardware/finn_enet_build.py <container>:/tmp/
docker exec <container> python /tmp/finn_enet_build.py
```

Reports land in
`/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/estimates_<timestamp>/report/`
inside the container.

## Running the real stitched-IP build (`finn_enet_ip_build.py`) — read this first

Unlike the estimate-only build above, `finn_enet_ip_build.py` actually shells
out to Vitis HLS and Vivado (per-node HLS/RTL synthesis, IP-XACT packaging,
stitching, rtlsim, OOC synthesis). This surfaced a container permission bug
that silently corrupts output if not fixed **before** the run.

### Fix the container's HOME permissions first (one-time per container)

`/home/thelegendiv` (the container user's home dir) is created **owned by
root** in this FINN image, even though the container runs as uid 1000
(`thelegendiv`). Vivado/Vitis need to write `~/.Xilinx/...` on every
invocation — including a small per-node Vivado sub-process that
`step_hw_ipgen` launches to package each synthesized node into IP-XACT
(`component.xml`). Without write access, every one of those sub-processes
fails with `ERROR: [Common 17-45] Cannot create directory`.

**The dangerous part**: `HLSSynthIP` does not check whether that packaging
actually succeeded — it only checks that C-synthesis completed — so the
build *looks* like it succeeded and checkpoints get saved, but every node's
IP directory is silently missing its real `component.xml`. This only
surfaces much later, as a confusing `ERROR: [BD 5-390] IP definition not
found for VLNV: ...` during `step_create_stitched_ip`'s `create_bd_cell`
call, potentially after hours of HLS synthesis.

Fix, once per container (needs no password — `docker exec -u root` uses
Docker's own access control, not the container user's credentials):

```bash
docker exec -u root <container> chown thelegendiv:thelegendiv /home/thelegendiv
```

Verify it worked before trusting any subsequent build:

```bash
docker exec <container> stat -c '%U:%G %a %n' /home/thelegendiv   # expect thelegendiv:thelegendiv 755
docker exec <container> sh -c "touch /home/thelegendiv/writetest && echo WRITE_OK && rm /home/thelegendiv/writetest"
```

If for some reason you can't get root in the container, the workaround is
to override `HOME` to a writable, bind-mounted directory instead (e.g.
`/tmp/finn_dev_thelegendiv/fakehome`) via `docker exec -e HOME=... `, but
this must be set for **every** process that touches Vivado/Vitis in that
run, including background/detached launches — easy to forget for one of
several `docker exec -d` calls.

### Don't trust a resumed checkpoint from a run that predates the HOME fix

FINN's native `start_step`/`stop_step` resume (`DataflowBuildConfig`, see
`finn_enet_ip_resume.py`) is genuine and works, but it only checks that a
node's IP output *directory* exists, not that packaging inside it actually
completed. If you resume from a checkpoint produced before the HOME fix
was applied (or under a different/inconsistent `HOME`), the resumed run
will silently reuse broken IP artifacts. **After changing the HOME fix,
start a fresh, non-resumed build** rather than resuming an old checkpoint.

### Model name argument is a filename stem, not a path

`finn_enet_ip_build.py [model_name]` looks for
`/home/thelegendiv/finn/notebooks/enet/<model_name>.onnx` — pass just the
stem (e.g. `quantEnet_finn_v1`), not a path or a `.onnx` suffix.

### Key config knobs

`target_fps` and `synth_clk_period_ns` in `cfg_stitched_ip` drive
`step_target_fps_parallelization`'s folding decisions — set these to your
actual throughput/clock requirement before a real run, not a placeholder
(the original estimate-only build's `target_fps=10000` above was exactly
that kind of placeholder, and is the direct cause of its huge LUT/DSP
overage).

### ZCU7EV LUT budget correction

The `XCZU7EV` resource dict in `finn_enet_ip_build.py` uses **230,400**
LUTs (real CLB LUT count per Xilinx DS891's detailed resource table), not
504,000 — that larger number is Xilinx's "System Logic Cells" marketing
metric from summary product-selection tables, a legacy 4-input-LUT-era
conversion (~2.1875×) that Vivado does **not** report utilization against.
Always cross-check hardcoded FPGA resource budgets against the detailed
per-device datasheet table, not a summary table.
