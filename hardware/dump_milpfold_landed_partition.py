"""Cheap (no-Vivado, no-HLS) landed-folding dump for one S12-dense 8-way partition,
generalized sibling of hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/
dump_milpfold_landed_partition2.py (parametrized partition index, any tag/preamble).

Re-derives the given partition exactly like finn_s12_build.py does up through
step_apply_folding_config (step_create_dataflow_partition_multi -> step_specialize_layers
-> GiveUniqueNodeNames/GiveReadableTensorNames -> step_target_fps_parallelization ->
GiveUniqueNodeNames (unprefixed, matches ApplyConfig's own renaming) -> step_apply_folding_config),
then re-runs AnnotateCycles so the saved onnx's cycles_estimate attribute reflects the just-applied
MILP folding. No HLS/ipgen/Vivado involved -- purely graph-transform + analytical cycle
annotation, seconds not hours.

Prerequisite: finn_s12_build.py <preamble_dir> --tag <tag> --conv-order <conv_order.json>
--folding-json <layer_bits_folding_tag.json> --bridge-only must have already run (produces
<output_dir>/hawq_folding_config_partition<N>.json).

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/dump_milpfold_landed_partition.py \\
        <preamble_dir> <bridged_folding_config.json> <partition_idx> <output_onnx_path>
"""
import os
import sys
import dataclasses

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

if len(sys.argv) < 5:
    print("Usage: dump_milpfold_landed_partition.py <preamble_dir> <bridged_folding_config.json> "
          "<partition_idx> <output_onnx_path>")
    sys.exit(1)
PREAMBLE_DIR = sys.argv[1]
FOLDING_CONFIG_FILE = sys.argv[2]
PARTITION_IDX = int(sys.argv[3])
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
    step_apply_folding_config,
)
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles  # noqa: E402

source_ckpt = os.path.join(PREAMBLE_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
print(f"Source checkpoint: {source_ckpt}")
print(f"Folding config file: {FOLDING_CONFIG_FILE}")
print(f"Partition: {PARTITION_IDX}")

flat_model = ModelWrapper(source_ckpt)
cfg = base.cfg_stitched_ip_partitioned_8way

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
print("Running: step_target_fps_parallelization")
kernel_model = step_target_fps_parallelization(kernel_model, cfg)
# same unprefixed re-rename step_apply_folding_config itself performs, done
# here explicitly first so our subsequent step call is a no-op rename (keeps
# behaviour identical to the real build / the bridge script).
kernel_model = kernel_model.transform(GiveUniqueNodeNames())

print(f"Running: step_apply_folding_config (MILP-bridged config from {FOLDING_CONFIG_FILE})")
cfg2 = dataclasses.replace(cfg, folding_config_file=FOLDING_CONFIG_FILE)
kernel_model = step_apply_folding_config(kernel_model, cfg2)

print("Running: AnnotateCycles (refresh cycles_estimate to reflect the MILP folding just applied)")
kernel_model = kernel_model.transform(AnnotateCycles())

os.makedirs(os.path.dirname(OUTPUT_ONNX) or ".", exist_ok=True)
kernel_model.save(OUTPUT_ONNX)
print(f"Saved landed (MILP-folded, no-HLS) partition {PARTITION_IDX} checkpoint: {OUTPUT_ONNX}")
