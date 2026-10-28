# dpu/ — Vitis AI 3.0 deployment on the ZCU104 DPUs

Alternative to the FINN path in [`../dataflow/`](../dataflow/README.md): the
trained network is quantized to INT8 with Vitis AI, compiled to an `.xmodel`
and run through VART on the DPU cores (two B4096) of the official ZCU104
Vitis AI 3.0 / 2022.2 board image. Nothing here reuses FINN/Brevitas exports.

## Where things run

| Piece | Location |
|---|---|
| This folder (versioned scripts) | repo, edited on Windows |
| `run_flow.sh` and all stages | **WSL** (`./run_flow.sh <job> <stage>`) |
| Vitis AI checkout | `/home/thelegendiv/Vitis-AI` (v3.0, reference only) |
| Container | `xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106` (`latest` is 3.5 — don't use) |
| Work area | `~/dpu_work/<job>/` on WSL ext4 (not `/mnt/c`: slow) |
| Inference | ZCU104 (`board_run.py`, python3 + VART) |

Source files stay in the repo; `run_flow.sh stage` copies the inputs a job
needs into the work area, and the container mounts that area as `/workspace`
and the job folder read-only as `/job`.

## One-time setup

1. `docker pull xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106`
   (if pulls fail with `x509: certificate signed by unknown authority`, fix
   the Docker/proxy CA trust, or `docker save`/`docker load` from another
   machine; do not disable TLS verification).
2. On the board: `xdputil query` — confirm 2 DPU cores, B4096, and note the
   fingerprint. Make sure `ARCH_JSON` (see `common.sh`) has the same
   fingerprint; if not, copy the matching `arch.json` into `arch/`.
3. Verify the container paths assumed in `common.sh` (`VAI_CONDA_SH`,
   `VAI_CONDA_ENV`, `ARCH_JSON`) after the first pull.

## New job

```
cp -r builds/_template builds/<job_name>   # name per the AGENTS.md convention
# edit job.env, model.py (plain PyTorch model + calibration data)
./run_flow.sh <job_name> setup
./run_flow.sh <job_name> inspect           # fix unsupported ops before going on
./run_flow.sh <job_name> all
```

Stages: `setup → stage → inspect → quantize → compile → deploy → bench`, each
logged to `~/dpu_work/<job>/logs/` and skipped on rerun once finished
(`FORCE=1` to redo).

## Benchmark table (Dice / FPS / latency / power)

After `compile` (the xmodel ends in an on-DPU argmax), run on the Windows host:

```
python dpu/collect_results.py [--job <job>] [--threads 2] [--repeats 5]
```

It copies the xmodel, `bench_board.py` and the 300 test images to the board (over
WSL ssh), runs the timed streams, single-image latency and the board power
measurement there, scores the class maps with the repo Dice, and upserts the
`stream_warm` / `single_image` rows (`int8_dpu`) into
`report/MILP/final_results/benchmark_s12_512.csv` + `_env.json`.
Timing covers DPU input → class map after argmax; preprocessing is done before
and excluded. Power follows the FINN flow (sum of the board's rail monitors, `deployment/FINN_interface/.../power_mon.c`):
the ZCU104 has only one INA226 (12 V input), but its two IRPS5401 regulators (i2c-4, 0x43/0x44) report `READ_POUT`
per output, so `bench_board.py` sums those rails (~100 Hz, idle vs. active). This is post-regulator (no conversion
losses, fan, PHY), about 8.2 W idle / 9.6 W active; the 12 V input INA226 (≈14.7 / 16.5 W) is sampled in the same sweep
and stored in the env JSON as `energy_12v_input`. The PMBus regulator at 0x13 returns no telemetry, so its rail is not
in the sum.
`--no-run` re-scores the stored outputs in `dpu/results/<job>/`.
`--qat` benchmarks the QAT model instead (`compiled_qat/`, results in `dpu/results/<job>_qat/`, `numeric_mode=int8_dpu_qat`).

## QAT fine-tuning (Vitis AI `QatProcessor`)

PTQ INT8 loses Dice vs FP32; QAT recovers it. The loop in `builds/<job>/qat_train.py` is standalone
(Vitis AI QAT needs torch 1.12, nnU-Net runs torch 2.0) but follows the repo's 15-epoch nnU-Net recipe:
250 train / 50 val iterations per epoch, batch 12, AdamW 1e-3 / wd 1e-2, poly(0.9) LR, grad-clip 12, nnU-Net
augmentation, CE + soft-Dice loss, seed 1234, starting from the FP32 checkpoint. `qat_model.py` rewrites the
ENet blocks at runtime (functional add/cat/interpolate, BN folded per branch in the initial block); the maths is
unchanged (calibrated Dice before training matches PTQ).

```
sbatch dpu/slurm/qat_12_dense_relu_warmstart150ep_dpu.job     # cluster GPU; env: dpu/slurm/setup_qat_env.sh
# copy <out>/{best,final} to ~/dpu_work/<job>/qat/ in WSL, then:
./run_flow.sh <job> qat_export       # QAT_TAG=best|final; evaluates + writes quant_qat/*_int.xmodel
./run_flow.sh <job> qat_compile      # + argmax -> compiled_qat/<job>_qat.xmodel (one DPU subgraph)
python dpu/collect_results.py --qat
```

Verified locally (CPU, 1 epoch smoke run): train → `to_deployable` → `qat_export.py` → `compile.sh` gives a single
DPU subgraph with argmax. Not yet verified: `setup_qat_env.sh` / GPU training on the cluster.

## Notes

- `model.py` is the only non-trivial piece: re-create the network with plain
  PyTorch ops (no Brevitas), transfer the weights, and verify it against the
  original float model before quantizing.
- Calibration preprocessing must match deployment preprocessing.
- The templates are untested skeletons written against the Vitis AI 3.0
  documentation; expect to adjust API details on first run.
- Single vs. dual core: `DPU_THREADS` (default 2) sets the number of runners;
  confirm real core utilization on the board rather than assuming it.
