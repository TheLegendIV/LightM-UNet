# S19 stage-partitioned stitched-IP build — running log

Living log for the effort to replace the monolithic (single-partition)
FINN stitched-IP build with a 5-way, stage-based partitioned build for
`quantEnet_s19_double_mid_int8`. Append new findings/decisions at the
bottom with a date. This is the log a fresh agent/session should read
first before touching anything under `hardware/finn_*partition*`,
`hardware/finn_enet_ip_build_partitioned.py`, or the FINN container.

## Why this exists

The original single-partition stitched-IP build (`finn_enet_ip_build_decomposed_prelu.py`)
showed near-quadratic Vivado `create_bd_cell` cost as cell count grew
(1665 cells → multi-day `CreateStitchedIP`, ~47.8GB peak Vivado RSS,
33+ hour estimated completion, OOM risk). Splitting the network into 5
independent partitions (cut at stage/down-up-sample boundaries) means each
partition's own stitch pays the quadratic cost on a much smaller n, and
partitions can be built concurrently since they share no state until a
final, cheap combining stitch.

## Current status (2026-08-15)

- Old monolithic Attempt 6 (PID 60949 / Vivado PID 90732, run inside the
  now-defunct container) was **killed** — was 94% through step 17/20
  (`step_set_fifo_depths`) after ~5 days, ~5.8GB RAM available, high OOM risk.
  No progress was recoverable regardless (see "Key findings" below).
- WSL2 memory ceiling raised from default (~50% of host RAM, ~46.9GiB) to
  **80GB** via `.wslconfig` (`C:\Users\<user>\.wslconfig`, `[wsl2] memory=80GB`).
  Also set `vmIdleTimeout=-1` to prevent the WSL2 VM auto-stopping when idle.
  **Any `.wslconfig` edit requires `wsl --shutdown` to take effect, which
  destroys ALL running WSL2 distros/containers** — do this deliberately,
  not casually.
- The FINN container was recreated (old one used `--rm`, so it was deleted
  when `wsl --shutdown` ran — don't use `--rm` for long-lived dev
  containers; use `--name <x>` instead so `docker start <x>` works later).
  Files under `/home/thelegendiv/finn/notebooks/enet/` survived the
  container's death (it's a bind mount from the host, not
  container-internal storage) — anything written only inside the
  container's writable layer would NOT have survived.
- New files written and verified to import/compile cleanly:
  - `hardware/finn_stage_partition.py` — detects the 5 stage boundaries
    structurally (asserts exactly 2 `StreamingMaxPool` + 5 `FMPadding_Pixel`
    nodes exist) and assigns `partition_id` 0–4 to every node.
  - `hardware/finn_partition_build_steps.py` — `step_create_dataflow_partition_multi`
    (no `assert len(sdp_nodes) == 1`), `step_build_all_partitions` (loops the
    existing generic per-node steps for each partition, optionally in
    parallel via `ProcessPoolExecutor`), `step_combine_partitions` (writes,
    but does not yet auto-run, a small top-level Vivado tcl that
    instantiates each partition's pre-packaged stitched IP and wires them
    in a linear chain via direct `connect_bd_intf_net`).
  - `hardware/finn_enet_ip_build_partitioned.py` — new top-level build
    script, modeled on `finn_enet_ip_build_decomposed_prelu.py`, wiring the
    above into a runnable `DataflowBuildConfig`.
- **First real end-to-end run launched** 2026-08-15 11:32 UTC container
  time, in the background (`nohup ... &`), log at
  `/tmp/finn_partitioned_build.log` (stdout/stderr) and the standard
  `build_dataflow.log` inside the run's own output dir. Using
  `step_build_all_partitions_capped` (`parallel=True, max_workers=3`) —
  capped below the partition count (5) to avoid CPU oversubscription,
  since each partition's own `step_hw_ipgen` internally parallelizes HLS
  synth via `NUM_DEFAULT_WORKERS=4`.
- Not yet reached: `assign_stage_partition_ids` / partition build / combine
  steps. `step_combine_partitions`'s generated tcl has **not yet been run
  through Vivado even once** — treat its first execution as a debug
  iteration, not a sure thing.

## Key findings (carried over, still true)

- **`step_set_fifo_depths` does a throwaway measurement stitch.**
  `InsertAndSetFIFODepths.apply()` calls `CreateStitchedIP` once internally
  with deliberately oversized FIFOs just to rtlsim-measure occupancy, then
  applies corrected depths and calls `reset_implementation` — meaning
  `step_create_stitched_ip` (or, in the partitioned flow, the final
  `CreateStitchedIP` call inside `_build_one_partition`) does a SEPARATE,
  real stitch afterward. Budget for stitching twice per partition.
- **Vivado only saves the project once, at the very end of its tcl script**
  (`save_bd_design`/`save_project` in `create_stitched_ip.py`, called after
  all `create_bd_cell` commands finish). A killed/crashed Vivado mid-script
  loses everything — no incremental persistence. This is a big part of why
  splitting into smaller partitions matters: less lost work per crash, and
  each individual stitch finishes fast enough that crash risk exposure per
  stitch is much lower.
- **`model.save()` only happens after a step function fully returns**
  (`build_dataflow.py`'s main loop) — `cfg.start_step`/`resolve_step_filename`
  let you resume from any COMPLETED step's checkpoint in
  `intermediate_models/`, but never from partway through a step.
- **FINN's DMA/DDR latency modeling is a no-op** (`IODMA_hls` inherits
  stub `get_exp_cycles`/`lut_estimation`/etc. that all return 0/None). Moot
  for this build — it has zero DMA nodes (confirmed via
  `build_dataflow_steps.py`'s step list).
- **`make_zynq_proj.py`'s `ZynqBuild.apply()` already implements a proven,
  working multi-partition loop** (`InsertFIFO → SpecializeLayers →
  GiveUniqueNodeNames → PrepareIP → HLSSynthIP → CreateStitchedIP` per
  `StreamingDataflowPartition`). The `assert len(sdp_nodes) == 1` blocker is
  only in the generic `build_dataflow_steps.py`'s `step_create_dataflow_partition`,
  not a fundamental FINN limitation — don't assume something is
  "unsupported" without checking whether a different FINN build flow
  already does it.
- **Interior partition-to-partition wiring needs no DMA** — `MakeZYNQProject`
  connects interior (non-graph-boundary) partitions via direct
  `connect_bd_intf_net` on `s_axis_%d`/`m_axis_%d`. DMA/AXI-lite/smartconnect
  is only used for the overall graph's first input / last output. This is
  why `step_combine_partitions` can be a simple linear-chain stitch (5 IP
  instances + 4 direct stream links), no DMA machinery needed.
- **No Vivado license blocker found** for running multiple concurrent
  Vivado sessions (no `LM_LICENSE_FILE`/`XILINXD_LICENSE_FILE`, no
  `lmstat`/`lmutil`, no license lines in `vivado.log`) — appears to be
  running on free/unlimited Vivado ML edition.

## Environment gotchas (see also `/memories/repo/finn-container-env.md`)

- **`docker exec <container> python3 ...` will fail to import
  finn/qonnx** unless you set `HOME=/tmp/home_dir` first (that's where the
  container's entrypoint script `pip install --user -e`'d everything —
  `docker exec` bypasses the entrypoint, so `HOME` defaults to
  `/home/thelegendiv` instead, where nothing is installed). Always run:
  `docker exec <container> bash -c "HOME=/tmp/home_dir python3 ..."`.
- **PowerShell mangles nested quotes** when running
  `docker exec ... bash -c "python3 -c '...'"` — multiple layers of quoting
  (PowerShell → bash -c → python -c) reliably break. Write the Python logic
  into a small script file, `docker cp` it in, and run
  `docker exec <container> bash -c "python3 /path/to/script.py"` instead.
- **Don't use `--rm`** when creating a long-lived FINN dev container — use
  `--name <x>` (and consider `--restart unless-stopped`) so it survives
  `wsl --shutdown`/host reboots via `docker start <x>`.
- **`.wslconfig` changes require `wsl --shutdown`**, which kills every
  running container in every WSL2 distro, not just the one you're working
  on. Confirm with the user before doing this — it is NOT reversible
  in-place (containers using `--rm` are deleted outright; others just
  stop and need `docker start`).
- Base pytorch/CUDA dev containers (unrelated `pytorch/pytorch:...` image
  used for the main LightM-UNet repo, not FINN) can silently die on
  startup if their entrypoint runs `apt-get install` and hits an
  interactive `tzdata` prompt — fix with `-e DEBIAN_FRONTEND=noninteractive`.

## Derived S19 partition boundaries (node index, out of 580 HW nodes)

Verified against the real intermediate checkpoint
`step_enet_convert_to_hw.onnx` (2 `StreamingMaxPool` at indices 7/81, 5
`FMPadding_Pixel` at indices 500/501/543/546/574 — index 574 is the
trailing `final` layer's own transpose, intentionally NOT a boundary):

| Partition | Name       | Node range  | Node count |
|-----------|------------|-------------|------------|
| 0         | initial    | [0, 7)      | 7          |
| 1         | stage1     | [7, 81)     | 74         |
| 2         | stage2/3   | [81, 500)   | 419 (dominant — 24/33 bottleneck blocks) |
| 3         | stage4     | [500, 543)  | 43         |
| 4         | stage5     | [543, 580)  | 37         |

Stage2/3 remains the long pole even after splitting — expect it alone to
take the majority of wall-clock time. The other 4 partitions should be
fast, which is the main argument for running them concurrently rather than
strictly sequentially.

## Bugs found & fixed during first real run (2026-08-15)

1. **`assign_stage_partition_ids` crashed at step 9/15** with
   `ValueError: Empty module name` inside `getCustomOp()`. Root cause: the
   original code called `getCustomOp(node)` on **every** node in the graph
   unconditionally, but not all nodes surviving `step_enet_convert_to_hw`
   are fpgadataflow HW ops — leftover `Reshape`/`Transpose`/`Constant`-type
   nodes have an empty `node.domain`, which crashes `getCustomOp`.
   `CreateDataflowPartition`'s own internal `assign_partition_id` lambda
   (`create_dataflow_partition.py`) already only looks at nodes with a
   `backend == "fpgadataflow"` attribute — everything else is left out of
   any partition (`-1`) regardless of what `partition_id` we'd have set.
   **Fix**: added `_is_fpgadataflow_node()` (checks the `backend` nodeattr
   via `get_by_name`) and skip `getCustomOp`/`set_nodeattr` for any node
   that isn't one — see `finn_stage_partition.py`.
2. Also discovered: `build_dataflow_cfg` drops into `pdb.post_mortem()` on
   any step exception. Since our runs are launched via `nohup ... &` with no
   TTY attached, `pdb` hits EOF on stdin and the process exits cleanly
   (confirmed no zombie/stuck process) — but this means log tails alone
   won't show a `(Pdb)` prompt as "hung", it'll just look like the process
   disappeared. Always check `ps aux` for the driver PID, not just log
   freshness, to distinguish "still running" from "crashed after printing
   its traceback".
3. Created `hardware/finn_enet_ip_resume_partitioned.py` (mirrors the
   existing `finn_enet_ip_resume.py` pattern for the single-partition
   build) to resume from a checkpoint via `cfg.start_step` instead of
   re-running the ~42-minute early tidy/streamline/convert-to-hw pipeline
   on every debug iteration. **Gotcha found while writing it**: the base
   build script (`finn_enet_ip_build_partitioned.py`) parses
   `sys.argv[1]`/`[2]` as `MODEL_NAME`/`FIFO_STRATEGY` at **import time**
   (module-level code) — if the resume script doesn't clear its own
   `sys.argv` (which holds `<failed_output_dir> <start_step>`) before
   importing the base module, the base module misreads those as
   `MODEL_NAME`/`FIFO_STRATEGY` (`ValueError: '<start_step>' is not a
   valid AutoFIFOSizingMethod`). Fixed by resetting `sys.argv = [sys.argv[0]]`
   before the import — relies on the resumed run having used the base
   script's default `MODEL_NAME`/`FIFO_STRATEGY` (true so far; needs a
   different fix if that ever changes).
4. First resume attempt (after fix #1) successfully reached
   `step_build_all_partitions_capped` [3/7] — confirming the resume
   mechanism works and that the `assign_stage_partition_ids` fix is
   correct. This is the first real exercise of the parallel
   per-partition build code (`ProcessPoolExecutor`, `max_workers=3`) —
   4 python processes observed running concurrently (1 main + 3 workers),
   consistent with the cap.
5. **Second crash**, inside `step_build_all_partitions_capped` itself:
   `PermissionError: [Errno 13] Permission denied:
   '/tmp/finn_dev_thelegendiv/code_gen_ipgen_FMPadding_rtl_0_...'` from
   `make_build_dir` during `PrepareIP` (HLS/RTL codegen). Root cause:
   `/tmp/finn_dev_thelegendiv` (FINN's shared per-user build-artifact dir,
   used across ALL steps) was owned `root:root` mode `0755` inside the
   `frosty_volhard` container, but our `docker exec` (no `-u` flag) runs
   as `uid=1000 (thelegendiv)`. `thelegendiv` isn't in the root group, so
   it can `rx` (list/traverse) but not `w` (create new entries) inside
   that dir. **This bug was latent from container creation** but only
   surfaces the first time a code path actually calls
   `make_build_dir`/`PrepareIP` — steps 1-8 (tidy/streamline/convert_to_hw)
   never call it, so it went unnoticed through the entire first ~42min
   run and the first resume attempt, only triggering once real per-node
   HLS codegen started inside `_build_one_partition`. **Fix**: one-time
   `docker exec -u root frosty_volhard bash -c "chown -R
   thelegendiv:thelegendiv /tmp/finn_dev_thelegendiv && chmod -R u+rwX
   /tmp/finn_dev_thelegendiv"`. If the container is ever recreated again,
   check/fix this ownership again before relaunching a build (add to a
   pre-flight checklist).
6. Second resume attempt (from `step_build_all_partitions_capped`, using
   the already-checkpointed `step_create_dataflow_partition_multi.onnx`)
   launched successfully after the permission fix, progressing past the
   `FMPadding_rtl_0` codegen point that crashed before, with many child
   processes spawned (HLS/vitis_hls synthesis workers) — currently in
   progress as of 2026-08-15 13:18.
7. **Partitions 0, 1, 3, 4 all completed successfully** (done at 13:20,
   15:42, 14:18, 14:54 respectively on 2026-08-15) — their per-node HLS/
   RTL codegen, folding, FIFO depth setting, and final `CreateStitchedIP`
   all worked correctly. Only `GenericPartition_2_` (stage2/3, the
   dominant partition — 419 raw nodes pre-folding, ~630+ after folding/
   FIFO-insertion adds many `StreamingFIFO_rtl_*`/`StreamingDataWidth
   Converter_rtl_*` nodes) remains.
8. **Severe Vivado scaling slowdown re-discovered in partition 2's final
   `CreateStitchedIP` pass** (`vivado_stitch_proj_3idblwdn/make_project.tcl`,
   PID 72083, started 2026-08-15 14:12) — **this is the SAME near-quadratic
   `create_bd_cell` cost documented above in "Why this exists" for the
   original 1665-cell monolithic build**, just recurring at a smaller (but
   still too-large) scale. Progress measured via
   `grep -c 'create_bd_cell -type module -reference' vivado.log`:
   - 13:2x-14:4x (2026-08-15): 264 -> 569 -> 626 cells (~60-90 cells/hour)
   - 2026-08-16 06:23 -> 13:25 (~7 hours later): 626 -> 634 cells (~1 cell/hour)
   The process is confirmed NOT hung (real CPU usage, ~35% avg, CPU-minutes
   climbing every poll) — genuinely still working, just impractically
   slowly. At this degraded rate, completing the remaining cells would
   take days to weeks. Vivado process memory has also climbed steadily
   (~5GB -> ~17.8GB peak) without matching cell-count growth in the same
   window, consistent with the same scaling problem (not corruption).
   **Conclusion: partition 2 (~630 cells) is still too large for a single
   flat `CreateStitchedIP` block design** — the 5-way split reduced the
   problem from 1665 cells to a max of ~630, which helped 4/5 partitions
   finish fine, but stage2/3 (419 raw nodes, containing 24/33 bottleneck
   blocks) is still over whatever the practical Vivado BD-scaling
   threshold is. **Options for next attempt**: (a) split stage2/3 further
   into 2+ sub-partitions (directly attacks root cause, most promising),
   (b) research a Vivado/FINN flag to avoid incremental
   validate/IP-catalog-rescan cost per `create_bd_cell` call, (c) let this
   run continue indefinitely and see if it eventually finishes (risky,
   low confidence). As of this writing the job has been left running
   (non-destructive) while this decision is made — **not yet killed**.

## Next steps / open items

- [ ] Watch `step_build_all_partitions_capped` [1/5 in this resume's step
      numbering] finish for all 5 partitions (currently in progress as of
      2026-08-15 13:18, resumed a 2nd time after the permission fix) —
      first real exercise of per-partition specialize/fold/codegen/ipgen/
      fifo/stitch + concurrent dispatch actually reaching HLS synthesis.
- [ ] **Inter-partition FIFOs are NOT auto-sized -- there are none.**
      `step_combine_partitions` connects adjacent partitions directly via
      `connect_bd_intf_net m_axis_0 -> s_axis_0` (see the function body),
      with no `StreamingFIFO` at the boundary. Auto-sizing
      (`step_set_fifo_depths`/`InsertAndSetFIFODepths`) only runs *inside*
      each partition during `_build_one_partition`. This is a known,
      deliberate simplification (adapted from `MakeZYNQProject`'s direct
      interior-node wiring) but is unvalidated for timing closure -- watch
      the OOC synthesis / rtlsim performance steps (which run after
      `step_combine_partitions`, unchanged) for any critical-path or
      backpressure surprises at partition boundaries. If timing fails at
      these boundaries, the fix would be inserting an explicit
      `StreamingFIFO` (even a shallow one, e.g. depth 2) between
      `m_axis_0`/`s_axis_0` in `step_combine_partitions`.
- [ ] `step_combine_partitions`'s generated tcl needs a first manual/real
      Vivado run — treat as a debug iteration.
- [ ] Decide whether `max_workers=3` is the right cap once real per-partition
      memory/CPU usage is observed (raise/lower based on actual behavior,
      not just the pre-run estimate).
- [ ] Update this log with the outcome of the first full run (success/failure,
      actual wall-clock time per partition, memory peak per partition, any
      bugs found in `step_combine_partitions`'s tcl generation).
- [ ] If more bugs are found in `step_build_all_partitions`/`_build_one_partition`,
      use `finn_enet_ip_resume_partitioned.py <output_dir> step_create_dataflow_partition_multi`
      (or an even earlier step) to resume rather than restarting from
      `step_qonnx_to_finn`.
