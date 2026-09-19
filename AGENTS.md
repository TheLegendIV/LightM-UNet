# AGENTS.md

Research repo compressing and deploying **LightM-UNet/ENet** (medical image
segmentation) onto a Zynq UltraScale+ **ZCU7EV** FPGA via Brevitas
quantization + the FINN dataflow compiler. Work here is iterative: train →
quantize (HAWQ) → export to FINN → build/synthesize → measure real hardware
resources → feed measurements back into the next config. Expect to resume
multi-day builds and cross-reference prior results before starting new ones.

## Directory map (see each folder's own README for details — don't duplicate here)

| Folder | Purpose | Docs |
|---|---|---|
| `enet/` | nnU-Net v2 fork + `ENet`/`QuantENet` model code, trainer classes | `enet/nnunetv2/**/readme.md` |
| `compression/` | Pruning/quantization sweep orchestration (Slurm), results tracking | [compression/README.md](compression/README.md), `compression/foundation_log.md` |
| `hardware/` | FINN export → HAWQ → build → OOC-synth pipeline for the active architecture | [hardware/README.md](hardware/README.md) |
| `deployment/` | Bitstreams (.bit/.xsa), drivers, PC↔FPGA image transfer interface | [deployment/image_transfer_interface/README.md](deployment/image_transfer_interface/README.md) |
| `analysis/501_ARCADE/` | Topology/quality metrics beyond Dice | [analysis/501_ARCADE/README.md](analysis/501_ARCADE/README.md) |
| `archive/` (any depth, e.g. `hardware/archive/`) | Retired code/docs kept for history | — |

**`**/archive/` is gitignored repo-wide.** New files created directly inside an
`archive/` dir will never show up in `git status`/get committed — that's
intentional (already-tracked files renamed into `archive/` stay tracked).

## The iteration loop and its naming convention

Every experiment is threaded through filenames rather than branches/subclasses:
`{arch}_{decoder/op-suffix}_{warmstartNNNep}_{alphaNNN}_{dummy|trained|finn_calibrated}`,
e.g. `12_dense_relu_warmstart150ep_alpha025_trained`. Before starting a new
variant, grep existing scripts for the closest matching suffix combination —
almost every pipeline stage (export → hawq dump → preamble → build → OOC
synth) already has a same-family sibling script to copy instead of writing
from scratch. Config identity in results tables is `(model_name, config)` —
reusing an existing name silently overwrites/upserts that row instead of
adding a new one.

- `compression/` sweeps: `config_name = {stage}_{descriptive-suffix}`
  (e.g. `stage2_U4_bnnative`), tracked in `compression/results.csv` via
  `collect_results.py` (idempotent — re-running a finished config is safe).
- `hardware/` builds: one script per pipeline phase per architecture variant
  (see [hardware/README.md](hardware/README.md) for the current S12-dense
  phase list). Results land in `hardware/results.csv` /
  `hardware/datasets/*.csv` via `hardware/collect_results.py`.
- Before a destructive change to generated artifacts, back up as
  `archive_pre_<YYYYMMDD>_<reason>/` next to the thing being changed (existing
  convention, not a script).

## Environments (three separate, non-interchangeable ones)

1. **Training** — Linux/Docker container built from `setup-enet.sh` /
   `requirements-enet-base.txt` (PyTorch 2.0.1+CUDA 11.7, Brevitas, `enet/`
   installed via `pip install -e .`). Runs `nnUNetv2_train` /
   `nnUNetv2_predict` (see `enet/nnunetv2/*/readme.md`).
2. **FINN build** — a separate, already-running Docker container (name
   changes across recreations, e.g. `practical_davinci` — find it with
   `docker ps`). FINN/QONNX/Brevitas are installed **editable under
   `/tmp/home_dir/.local/...`, not any conda/global site-packages**, because
   the container's entrypoint sets `HOME=/tmp/home_dir` before `pip install
   -e`. Plain `docker exec <container> python3 script.py` uses the wrong
   `HOME` and fails with `ModuleNotFoundError: No module named 'qonnx'`.
   **Always** run FINN/QONNX code as:
   `docker exec -e HOME=/tmp/home_dir <container> python3 <script>`.
   Scripts are deployed by `docker cp`-ing individual `.py` files into the
   container's flat `/home/thelegendiv/finn/notebooks/enet/` — this repo's
   local `hardware/` subfolder layout is git-organization only and doesn't
   need to mirror the container.
   Vivado/Vitis HLS 2022.2 live at `/tools/Xilinx/{Vivado,Vitis_HLS}/2022.2`
   in-container but aren't on `PATH` — `source .../settings64.sh` first.
3. **Host (Windows)** — for `compression/hawq/*.py` / `hardware/collect_results.py`,
   which need torch+pulp+the vendored `enet/nnunetv2` package but have no
   venv/conda set up in this repo. Default Windows `python`/`py` is 3.6 with
   no torch.

## Known sharp edges

- PowerShell mangles nested double quotes in
  `docker exec ... bash -c "python3 -c '...'"`. Write the snippet to a temp
  `.py` file, `docker cp` it in, then `docker exec ... bash -c "python3 /path/script.py"`.
- A long-running command piped through `| tail -N` in the terminal tool shows
  **zero** output until the whole pipe hits EOF — don't assume it's stuck;
  poll via a separate `docker exec`/file-mtime check instead.
- When a build container disappears, check the WSL bind-mounted host path
  (`finn/notebooks/enet/...`, persists) before assuming results are lost —
  only files under the container's ephemeral `/tmp/finn_dev_<user>/...` are
  actually at risk.
- `FINN/` (repo root) is an intentionally empty placeholder — the real FINN
  toolchain lives only inside the build container.

For accumulated build-specific bugs, resource-model corrections, and per-build
status, check this workspace's Copilot repo memory (`finn_gotchas.md`,
`finn-container-env.md`, `finn_enet_build_fixes.md`) before re-deriving a fix
from scratch.
