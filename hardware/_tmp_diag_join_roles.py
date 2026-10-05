"""Throwaway diagnostic: reproduce build_partition_folding_config's role_by_name construction for ONE
partition and print everything relevant to the 'dup transition' (<block>.out_act -> <next>.dup)
unresolved-producer mystery -- specifically len(adds) vs len(blocks) in _find_join_thresholds, and
whether 'down1.out_act' lands in role_by_name at all.

Usage (in container): python3 _tmp_diag_join_roles.py <partition_idx>
"""
import dataclasses
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

PREAMBLE_DIR = ("/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
                "S12_dense_256_u4_analytical_v1_finn_calibrated_rtl_mvau_256x256_composed_preamble_20261005_164026")
CONV_ORDER = "/home/thelegendiv/finn/notebooks/enet/quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_conv_order.json"
FOLDING_JSON = "/home/thelegendiv/finn/notebooks/enet/layer_bits_folding_S12_dense_256_u4_analytical_v1_final.json"
PROBE_OUT = "/tmp/_tmp_diag_join_roles_out"
os.makedirs(PROBE_OUT, exist_ok=True)
os.environ["FINN_BUILD_DIR"] = os.path.join(PROBE_OUT, "finn_build_tmp")
os.makedirs(os.environ["FINN_BUILD_DIR"], exist_ok=True)

idx = int(sys.argv[1])

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402

_real_argv, sys.argv = sys.argv, sys.argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from finn.builder.build_dataflow_steps import step_specialize_layers, step_target_fps_parallelization  # noqa: E402
from finn_partition_build_steps import step_create_dataflow_partition_multi  # noqa: E402
from finn_s12_build_steps import (  # noqa: E402
    install_relaxed_stage_boundaries, build_partition_role_nodes, _find_join_thresholds,
    _find_block_join_nodes, _previous_block_name, load_partition_logical_names,
)

install_relaxed_stage_boundaries()

cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=PROBE_OUT)

with open(FOLDING_JSON) as f:
    folding_block = json.load(f)
per_layer = folding_block["per_layer"]

flat_ckpt = os.path.join(PREAMBLE_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
parent = step_create_dataflow_partition_multi(ModelWrapper(flat_ckpt), cfg)
sdp_nodes = parent.get_nodes_by_op_type("StreamingDataflowPartition")
assert len(sdp_nodes) == 8, len(sdp_nodes)
partition_model_fn = getCustomOp(sdp_nodes[idx]).get_nodeattr("model")

logical = load_partition_logical_names(PREAMBLE_DIR, CONV_ORDER)
logical_names = logical[idx][0]
print(f"partition {idx}: {len(logical_names)} logical_names, first 5: {logical_names[:5]}")

kernel_model = ModelWrapper(partition_model_fn)
kernel_model = step_specialize_layers(kernel_model, cfg)
kernel_model = kernel_model.transform(GiveUniqueNodeNames(sdp_nodes[idx].name + "_"))
kernel_model = kernel_model.transform(GiveReadableTensorNames())
kernel_model = step_target_fps_parallelization(kernel_model, cfg)
kernel_model = kernel_model.transform(GiveUniqueNodeNames())

prev_block = _previous_block_name(CONV_ORDER, logical_names[0]) if logical_names else None
print(f"prev_block = {prev_block!r}")

adds = [n for n in kernel_model.graph.node if n.op_type.startswith("AddStreams")]
blocks = [ln[: -len(".expand.0")] for ln in logical_names if ln.endswith(".expand.0")]
print(f"len(adds) = {len(adds)}  adds = {[n.name for n in adds]}")
print(f"len(blocks) = {len(blocks)}  blocks = {blocks}")

joins = _find_join_thresholds(kernel_model, logical_names, prev_block)
print(f"\n_find_join_thresholds -> {len(joins)} entries:")
for node_name, (block, kind) in joins.items():
    print(f"  {node_name:30s} -> ({block}, {kind})")

join_roles = _find_block_join_nodes(kernel_model, logical_names, prev_block)
print(f"\n_find_block_join_nodes -> {len(join_roles)} roles:")
for role in sorted(join_roles):
    if "down1" in role or "regular1.0" in role:
        print(f"  {role:30s} -> {join_roles[role]}")

role_by_name = build_partition_role_nodes(kernel_model, logical_names, per_layer)
role_by_name.update(join_roles)
print(f"\nFinal role_by_name has {len(role_by_name)} entries. down1.* / regular1.0.dup entries:")
for role in sorted(role_by_name):
    if role.startswith("down1.") or role == "regular1.0.dup":
        print(f"  {role:30s} -> {role_by_name[role]}")
print("\n'down1.out_act' in role_by_name:", "down1.out_act" in role_by_name)
print("'down1.thr_out' in role_by_name:", "down1.thr_out" in role_by_name)
