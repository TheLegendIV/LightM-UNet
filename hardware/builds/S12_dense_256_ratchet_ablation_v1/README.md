# S12_dense_256_ratchet_ablation_v1 (hardware side)

Partition-2 OOC builds for the DSR (formerly "ratchet") ablation of the 256x256 S12 dense nearest-upsample (noconv) ReLU net, uniform INT6. The MILP arms live in
`MILP/artifacts/S12_dense_256_ratchet_ablation_v1/` (see its README for the arms, the DSR rule and the results). Written but NOT launched: the FINN container is run by hand.

## What gets built (partition 2, one build per distinct MILP folding + the control)

| build tag | folding | for arms |
|---|---|---|
| `ratchet_1pct` | MILP folding A | ratchet_1pct |
| `ratchet_25pct` | MILP folding B | ratchet_25pct |
| `ratchet_100pct` | MILP folding C | ratchet_100pct |
| `ratchet_200pct` | MILP folding D | ratchet_200pct |
| `ratchet_off` | MILP folding E | ratchet_off |
| `analytical_25pct` | analytical-flow folding F (net_fold.py), FIFO depths forced from the analytical simulation | analytical_25pct |
| `analytical_25pct_finnfifo` | same folding F, folding json WITHOUT the FIFO lists (the bridge prints "no inter_block_fifos/intra_block_fifos" and forces nothing): FINN's rtlsim FIFO autosizer decides every depth | analytical_25pct_finnfifo |
| `ratchet_<arm>_simfifo` (5 builds) | the SAME foldings A-E as the MILP arms, FIFO lists replaced by the simulated ones (`MILP/analytical/net_explicit.py`, every FIFO of the block simulation, grown until the whole-net chain passes) | ratchet_<arm>_simfifo |
| `ratchet_ablation_finn_autofold` | none: FINN `step_target_fps_parallelization`, `--target-fps 250 --mvau-wwidth-max 72` | FINN auto-fold control |

`arms_to_build.txt` now lists ONLY the five `ratchet_<arm>_simfifo` builds (the other rows were built earlier; the control is rebuilt only with `WITH_CONTROL=1`). The list comes from `MILP/artifacts/S12_dense_256_ratchet_ablation_v1/arms_to_build.txt` (rerun `summarize_arms.py` after re-solving: if two arms stop sharing a folding they get their own build).
Partition 2 = stage2.0 .. stage2.4 (dilations 2, 4, 8, 16, 2); confirm with the conv order before trusting the label.

## Inputs (docker cp into the flat `/home/thelegendiv/finn/notebooks/enet/`)

* `hardware/builds/S12_dense_256_u4_analytical_v1/outputs/quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep.onnx` (+ `.onnx.data`) - the QAT'd export (use `..._finn_calibrated.onnx` for the PTQ one and change `MODEL` in the script).
* `.../outputs/quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_conv_order.json` (shared by both exports).
* `MILP/artifacts/S12_dense_256_ratchet_ablation_v1/<arm>/layer_bits_folding_<arm>.json` for every arm in `arms_to_build.txt`, and `arms_to_build.txt` itself.
* The active build scripts: `hardware/finn_s12_preamble.py`, `finn_s12_build.py`, `finn_s12_build_steps.py`, `dump_milpfold_landed_partition.py`, `checks/check_milp_vs_landed_folding.py`
  (plus the modules they import, already in the container if the S12 flow ran before).
* `run_arms.sh` from this folder.

## Run

```
docker exec -e HOME=/tmp/home_dir <container> bash -c 'cd /home/thelegendiv/finn/notebooks/enet && STEP=bridge bash run_arms.sh 2>&1 | tee /tmp/ratchet256_bridge.log'   # no Vivado: preamble, bridge, landed-folding gate
docker exec -e HOME=/tmp/home_dir <container> bash -c 'cd /home/thelegendiv/finn/notebooks/enet && nohup bash run_arms.sh > /tmp/ratchet256_run.log 2>&1 &'                  # + queue the builds (max 4 concurrent)
```

Read the first run before launching builds:
* `FIFO role bridge: N matched / M unresolved` per arm (`/tmp/ratchet256_bridge_<arm>.log`): the MILP's `intra_block_fifos` (skip and prefetch FIFOs) and `inter_block_fifos` (depth 2) are forced by role;
  everything unmatched keeps FINN's autosized depth (see `fifo_force_report_partition_2.json` after a build).
* `LANDED CHECK: OK` - the MILP (PE, SIMD, thr_pe, SWU) folding landed unchanged in partition 2.

## Collect

Copy `finn_deployment_outputs/*_partition2_*` out of the container, then
`python3 hardware/builds/S12_dense_256_ratchet_ablation_v1/collect_results_partitions.py --deployment-root <copy> --arm-map MILP/artifacts/S12_dense_256_ratchet_ablation_v1/arm_build_map.csv [--dsp-overrides-json real_dsp.json]`
upserts one row per arm (identical foldings share the build and say so in `notes`) into `hardware/results.csv` as `model_name=S12_dense_256_u4_analytical_v1`, `config=<arm>_milpfold_partition_2_ooc_synth`
(control: `ratchet_ablation_finn_autofold_autofold_partition_2_ooc_synth`). FINN's DSP field is unreliable (known parser bug): pass the real DSP counts from each build's `vivado.log` via `--dsp-overrides-json`
(`{"<tag>_milpfold": 123, ...}`).

## Caveats

* `finn_s12_build.py --fifo-autosize` defaults to `fixed2` (every FIFO starts at depth 2, no rtlsim) -- unmatched
  FIFOs on the 6 MILP-folded arms therefore stay at depth 2, NOT a large autosized depth. Only
  `ratchet_ablation_finn_autofold` and `analytical_25pct_finnfifo` pass `--fifo-autosize rtlsim` explicitly in
  `run_arms.sh` (both have no `inter_block_fifos`/`intra_block_fifos` to force, so they need FINN's real
  `largefifo_rtlsim` autosizer or every FIFO silently stays at depth 2 -- this exact gap caused
  `analytical_25pct_finnfifo`'s prior deadlock and was NOT a FINN autosizer bug). Keep this in mind before
  comparing BRAM across arms: the 6 fixed2 arms' unmatched-FIFO BRAM is near-zero by construction, the 2
  rtlsim arms' is real/measured.
* The bridge does not apply the argmax PE (`final.argmax`) and the FMPadding_Pixel SIMD: irrelevant for partition 2, relevant for partitions 5-7.
* `hardware/README.md` and `hardware/builds/README.md` describe an older flow; this folder describes the unified `finn_s12_*` flow.
* Nothing here has been run: `bash -n` and `py_compile` only.
