#!/usr/bin/env bash
# Vitis AI 3.0 -> ZCU104 DPU flow. Run inside WSL:
#
#   ./run_flow.sh <job> <stage>        # one stage
#   ./run_flow.sh <job> all            # setup..bench, skipping finished stages
#   FORCE=1 ./run_flow.sh <job> quantize
#
# Stages (each resumable via $DPU_WORK_ROOT/<job>/.done/<stage>):
#   setup     check container image + board reachability, print DPU info
#   stage     copy checkpoint/calibration/arch inputs into the WSL work dir
#   inspect   Vitis AI Model Inspector against $INSPECT_TARGET   (container)
#   quantize  INT8 calibration, evaluation, xmodel export        (container)
#   compile   vai_c_xir against $ARCH_JSON                       (container)
#   deploy    copy compiled model + board_run.py to the board
#   bench     run board_run.py with $DPU_THREADS runners, fetch results
#
# QAT (trained on the cluster, see slurm/): put the result in $DPU_WORK_ROOT/<job>/qat/{best,final}/, then
#   qat_export    evaluate + export QAT_TAG (best|final, default best)  (container)
#   qat_compile   compile it into compiled_qat/                         (container)
#   COMPILED=compiled_qat ./run_flow.sh <job> deploy|bench
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 2; }
[[ $# -eq 2 ]] || usage
load_job "$1"; STAGE="$2"

run_stage() {
    local name="$1"; shift
    if stage_done "$name"; then log "skip $name (done; FORCE=1 to rerun)"; return; fi
    log "stage: $name"
    "$@" 2>&1 | tee "$JOB_WORK/logs/$name.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || die "stage $name failed (see $JOB_WORK/logs/$name.log)"
    mark_done "$name"
}

stage_setup() {
    docker image inspect "$VAI_IMAGE" >/dev/null 2>&1 \
        || die "image $VAI_IMAGE missing: docker pull $VAI_IMAGE"
    [[ -n "$BOARD_IP" ]] || die "BOARD_IP not set (job.env or environment)"
    log "board DPU info (expect 2 cores, B4096):"
    board_ssh "xdputil query"
    log "Check the fingerprint above matches $ARCH_JSON, then copy that arch.json to dpu/arch/ if it differs."
}

stage_stage() {
    # CHECKPOINT and CALIB_DIR are defined in builds/<job>/job.env.
    [[ -f "${CHECKPOINT:-}" ]] || die "CHECKPOINT not found: ${CHECKPOINT:-<unset>}"
    cp -u "$CHECKPOINT" "$JOB_WORK/inputs/checkpoint.pth"
    if [[ -d "${CALIB_DIR:-}" ]]; then
        rsync -a --delete "$CALIB_DIR"/ "$JOB_WORK/inputs/calib/"
    else
        log "WARN: CALIB_DIR not found (${CALIB_DIR:-<unset>}); fine for inspect, required for quantize"
    fi
    if [[ -d "${LABEL_DIR:-}" ]]; then
        rsync -a --delete "$LABEL_DIR"/ "$JOB_WORK/inputs/labels/"
    fi
    [[ -d "$DPU_DIR/arch" ]] && rsync -a "$DPU_DIR/arch"/ "$JOB_WORK/inputs/arch/"
}

stage_inspect()  { in_container "python /job/inspect_model.py"; }
stage_quantize() {
    in_container "python /job/quantize.py calib"
    in_container "python /job/quantize.py test"
}
stage_compile()  { in_container "bash /job/compile.sh"; }

QAT_TAG="${QAT_TAG:-best}"
COMPILED="${COMPILED:-compiled}"
stage_qat_export() {
    in_container "python /job/qat_export.py --src /workspace/qat/$QAT_TAG --out /workspace/quant_qat"
}
stage_qat_compile() {
    in_container "QUANT_DIR=/workspace/quant_qat COMPILE_OUT=/workspace/compiled_qat COMPILE_NAME=${JOB}_qat bash /job/compile.sh"
}

stage_deploy() {
    [[ -n "$BOARD_IP" ]] || die "BOARD_IP not set"
    board_ssh "mkdir -p $BOARD_DIR"
    board_scp "$JOB_WORK/$COMPILED"/*.xmodel "$JOB_SRC/board_run.py" \
        "$BOARD_USER@$BOARD_IP:$BOARD_DIR/"
}

stage_bench() {
    local model
    model="$(basename "$(ls "$JOB_WORK/$COMPILED"/*.xmodel | head -n1)")"
    board_ssh "cd $BOARD_DIR && python3 board_run.py --model $model --threads $DPU_THREADS ${BENCH_ARGS:-}"
    mkdir -p "$JOB_WORK/board_out"
    board_scp "$BOARD_USER@$BOARD_IP:$BOARD_DIR/out/*" "$JOB_WORK/board_out/" || true
}

case "$STAGE" in
    setup|stage|inspect|quantize|compile|qat_export|qat_compile|deploy|bench)
        run_stage "$STAGE" "stage_$STAGE" ;;
    all)
        for s in setup stage inspect quantize compile deploy bench; do
            run_stage "$s" "stage_$s"
        done ;;
    *) usage ;;
esac
