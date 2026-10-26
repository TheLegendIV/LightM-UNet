#!/usr/bin/env bash
# Run the Vitis AI QAT fine-tune on this PC's CPU (about 6-7 h for the full 15-epoch recipe), in a detached
# container, so it survives closing the terminal. Resumable: rerun to continue from ckpt_latest.pth.
#
#   ./run_qat_local.sh [job]          # start (default job 12_dense_relu_warmstart150ep_fp32)
#   docker logs -f qat_dpu            # follow;  docker stop qat_dpu  # pause (resume by rerunning)
#
# Output: $DPU_WORK_ROOT/<job>/qat/{best,final}/ (+ ckpt_*.pth, history.json), ready for `run_flow.sh <job> qat_export`.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_job "${1:-12_dense_relu_warmstart150ep_fp32}"

DATA="${QAT_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/nnUNet_preprocessed/Dataset509_ARCADE_1x1_4c}"
[[ -d "$DATA/nnUNetPlans_2d" ]] || die "preprocessed data not found: $DATA/nnUNetPlans_2d (set QAT_DATA)"
[[ -f "$JOB_WORK/inputs/checkpoint.pth" ]] || die "run './run_flow.sh $JOB stage' first (needs inputs/checkpoint.pth)"
docker ps -a --format '{{.Names}}' | grep -qx qat_dpu && die "container qat_dpu exists: docker rm -f qat_dpu (or docker start qat_dpu)"

mkdir -p "$JOB_WORK/qat" "$JOB_WORK/qat_pylibs"
docker run -d --name qat_dpu --network=host --shm-size=8g \
    -v "$JOB_WORK":/workspace \
    -v "$DATA":/data:ro \
    -v "$JOB_SRC":/job:ro \
    -w /workspace "$VAI_IMAGE" bash -c "
source $VAI_CONDA_SH && conda activate $VAI_CONDA_ENV
export PYTHONPATH=/workspace/qat_pylibs:\$PYTHONPATH
[ -d /workspace/qat_pylibs/batchgenerators ] || pip install --no-deps --target /workspace/qat_pylibs batchgenerators==0.25 future unittest2 threadpoolctl pandas
python /job/qat_train.py --data /data/nnUNetPlans_2d --fp32-checkpoint /workspace/inputs/checkpoint.pth \
    --out /workspace/qat --epochs 15 --iters 250 --val-iters 50 --batch-size 12 --lr 1e-3 \
    --workers 6 --device cpu --checkpoint-every 1 2>&1 | tr '\r' '\n' | grep -a --line-buffered -v -E 'VAIQ|OpInfo|it/s|^ *\|? *$'
"
log "started container qat_dpu; follow with: docker logs -f qat_dpu"
