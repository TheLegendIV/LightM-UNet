"""Cheap (no-Vivado, no-HLS) FINN-autofold-only landed dump for partition 2 --
the control variant for the S12_dense_dsr_ablation_v1 MILP sweep. Lets FINN's
own SetFolding choose PE/SIMD (step_target_fps_parallelization only, no
step_apply_folding_config) at the SAME target_fps/mvau_wwidth_max the MILP
solve used (target_fps=305.17, mvau_wwidth_max=80 per
MILP/artifacts/S12_dense_dsr_ablation_v1/run_ablation.sh), then saves a landed
checkpoint with cycles_estimate annotated -- same recipe as
dump_milpfold_landed_partition2.py minus the folding-config override.

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/dump_autofold_landed_partition2.py \\
        <hawq_preamble_output_dir> <target_fps> <mvau_wwidth_max> <output_onnx_path>
"""
import os
import sys
import dataclasses

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

if len(sys.argv) < 5:
    print("Usage: dump_autofold_landed_partition2.py <preamble_dir> <target_fps> <mvau_wwidth_max> <output_onnx_path>")
    sys.exit(1)
PREAMBLE_DIR = sys.argv[1]
TARGET_FPS = float(sys.argv[2])
MVAU_WWIDTH_MAX = int(sys.argv[3])
OUTPUT_ONNX = sys.argv[4]

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from finn_partition_build_steps import step_create_dataflow_partition_multi  # noqa: E402
from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_specialize_layers,
    step_target_fps_parallelization,
)
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles  # noqa: E402

PARTITION_IDX = 2

source_ckpt = os.path.join(PREAMBLE_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
print(f"Source checkpoint: {source_ckpt}")
print(f"target_fps={TARGET_FPS} mvau_wwidth_max={MVAU_WWIDTH_MAX}")

flat_model = ModelWrapper(source_ckpt)
cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, target_fps=TARGET_FPS, mvau_wwidth_max=MVAU_WWIDTH_MAX)

print("Running step_create_dataflow_partition_multi (re-split, deterministic)...")
parent_model = step_create_dataflow_partition_multi(flat_model, cfg)
sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
sdp_node = sdp_nodes[PARTITION_IDX]
sdp_inst = getCustomOp(sdp_node)
partition_model_fn = sdp_inst.get_nodeattr("model")
print(f"Partition {PARTITION_IDX} -> {sdp_node.name} -> {partition_model_fn}")

prefix = sdp_node.name + "_"
kernel_model = ModelWrapper(partition_model_fn)
print(f"Loaded raw partition {PARTITION_IDX} model: {len(kernel_model.graph.node)} nodes")

print("Running: step_specialize_layers")
kernel_model = step_specialize_layers(kernel_model, cfg)
kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
kernel_model = kernel_model.transform(GiveReadableTensorNames())
print("Running: step_target_fps_parallelization (auto-fold, no MILP override)")
kernel_model = step_target_fps_parallelization(kernel_model, cfg)

print("Running: AnnotateCycles")
kernel_model = kernel_model.transform(AnnotateCycles())

os.makedirs(os.path.dirname(OUTPUT_ONNX) or ".", exist_ok=True)
kernel_model.save(OUTPUT_ONNX)
print(f"Saved landed (FINN-autofold, no-HLS) partition {PARTITION_IDX} checkpoint: {OUTPUT_ONNX}")
