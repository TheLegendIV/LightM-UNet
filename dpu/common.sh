#!/usr/bin/env bash
# Shared settings/helpers for dpu/run_flow.sh. Meant to be sourced, run inside WSL.
# Override any value from the environment or from builds/<job>/job.env.

DPU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Vitis AI container ---------------------------------------------------
# Pinned Vitis AI 3.0 image (the `latest` tag is 3.5 -- do not use it).
VAI_IMAGE="${VAI_IMAGE:-xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106}"
VAI_CONDA_SH="${VAI_CONDA_SH:-/opt/vitis_ai/conda/etc/profile.d/conda.sh}"  # verify after first pull
VAI_CONDA_ENV="${VAI_CONDA_ENV:-vitis-ai-pytorch}"

# --- Work area (WSL ext4, NOT /mnt/c: calibration I/O is slow there) ------
DPU_WORK_ROOT="${DPU_WORK_ROOT:-$HOME/dpu_work}"

# --- Board (ZCU104, official Vitis AI 3.0 / 2022.2 image) -----------------
BOARD_IP="${BOARD_IP:-}"                 # set in builds/<job>/job.env or environment
BOARD_USER="${BOARD_USER:-root}"
BOARD_DIR="${BOARD_DIR:-/home/root/dpu_deploy}"
DPU_THREADS="${DPU_THREADS:-2}"          # one VART runner per DPU core

# --- Compile target --------------------------------------------------------
# Must match the fingerprint reported by `xdputil query` on the board.
ARCH_JSON="${ARCH_JSON:-/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json}"
INSPECT_TARGET="${INSPECT_TARGET:-DPUCZDX8G_ISA1_B4096}"  # or a 0x... fingerprint

log()  { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

load_job() {
    JOB="${1:?job name required}"
    JOB_SRC="$DPU_DIR/builds/$JOB"
    [[ -d "$JOB_SRC" ]] || die "no such job dir: $JOB_SRC"
    [[ -f "$JOB_SRC/job.env" ]] && source "$JOB_SRC/job.env"
    JOB_WORK="$DPU_WORK_ROOT/$JOB"
    mkdir -p "$JOB_WORK"/{inputs,float,inspect,quant,compiled,logs,.done}
}

# Stage markers make the flow resumable; FORCE=1 reruns everything.
stage_done()  { [[ -z "${FORCE:-}" && -f "$JOB_WORK/.done/$1" ]]; }
mark_done()   { touch "$JOB_WORK/.done/$1"; }

# Run a command inside the Vitis AI container with the job's work dir at
# /workspace and the job's sources (read-only) at /job. Mirrors the mounts and
# user env of Vitis-AI/docker_run.sh, but non-interactive and without the
# license prompt.
in_container() {
    docker run --rm --network=host \
        -v /dev/shm:/dev/shm \
        -v "$JOB_WORK":/workspace \
        -v "$JOB_SRC":/job:ro \
        -v "$DPU_DIR":/dpu_common:ro \
        -e USER="$(whoami)" -e UID="$(id -u)" -e GID="$(id -g)" \
        -e JOB="$JOB" \
        -e INPUT_SHAPE="${INPUT_SHAPE:-}" \
        -e INSPECT_TARGET="$INSPECT_TARGET" \
        -e ARCH_JSON="$ARCH_JSON" \
        -w /workspace \
        "$VAI_IMAGE" \
        bash -c "source $VAI_CONDA_SH && conda activate $VAI_CONDA_ENV && $*"
}

board_ssh() { ssh "$BOARD_USER@$BOARD_IP" "$@"; }
board_scp() { scp -O -r "$@"; }  # -O: the board's dropbear has no sftp-server
