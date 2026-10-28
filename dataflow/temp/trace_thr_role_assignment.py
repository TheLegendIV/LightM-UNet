"""Instrument build_partition_folding_config's role-resolution calls to pin down EXACTLY which
code path sets by_name["down2.thr_s"] / by_name["down2.thr_r"] and to what node, and in what order.
Does NOT run the (expensive) CreateStitchedIP/rtlsim portion -- only the folding-config + fifo_plan
construction, matching rebuild_partition1_refix2_and_rtlsim.py's setup exactly.
"""
import dataclasses
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
os.environ.setdefault("FINN_ROOT", "/home/thelegendiv/finn")

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
BUILD_TAG = "S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_refix2"
os.environ["FINN_BUILD_DIR"] = os.path.join(ENET_DIR, "finn_build_tmp", BUILD_TAG + "_trace")
FINN_ROOT = "/home/thelegendiv/finn"

PREAMBLE_DIR = os.path.join(
    ENET_DIR, "finn_deployment_outputs", "S12_dense_256_u4_analytical_v1_ft15ep_preamble_20261005_220558"
)
CONV_ORDER = os.path.join(ENET_DIR, "quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_conv_order.json")
FOLDING_JSON = os.path.join(
    ENET_DIR, "layer_bits_folding_S12_dense_256_u4_analytical_v1_int6_fps250_lat200_20261006.json"
)
PART_IDX = 1

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402

import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
from finn_partition_build_steps import step_create_dataflow_partition_multi  # noqa: E402
import finn_s12_build_steps as bs  # noqa: E402
from finn_s12_build_steps import (  # noqa: E402
    build_partition_folding_config, install_relaxed_stage_boundaries, load_partition_logical_names,
)

install_relaxed_stage_boundaries()

# ---- instrumentation: wrap the 3 candidate functions that can touch down2.thr_s/thr_r ----
_orig_fft = bs._find_following_thresholding
_orig_skip = bs._find_dn_block_via_skip
_orig_join = bs._find_join_thresholds


def traced_fft(kernel_model, node):
    r = _orig_fft(kernel_model, node)
    print(f"[TRACE _find_following_thresholding] node={node.name} -> {r.name if r is not None else None}", flush=True)
    return r


def traced_skip(kernel_model, skip_mvau_node):
    r = _orig_skip(kernel_model, skip_mvau_node)
    print(f"[TRACE _find_dn_block_via_skip] skip_mvau_node={skip_mvau_node.name} -> {r}", flush=True)
    return r


def traced_join(kernel_model, logical_names, prev_block=None, claimed_mvau_names=None):
    r = _orig_join(kernel_model, logical_names, prev_block, claimed_mvau_names)
    print(f"[TRACE _find_join_thresholds] claimed_mvau_names_down2_related="
          f"{[n for n in (claimed_mvau_names or []) ]}", flush=True)
    print(f"[TRACE _find_join_thresholds] result={r}", flush=True)
    return r


bs._find_following_thresholding = traced_fft
bs._find_dn_block_via_skip = traced_skip
bs._find_join_thresholds = traced_join


def main():
    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir="/tmp/trace_out")
    os.makedirs("/tmp/trace_out", exist_ok=True)

    flat_ckpt = os.path.join(PREAMBLE_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
    parent = step_create_dataflow_partition_multi(ModelWrapper(flat_ckpt), cfg)
    sdp_nodes = parent.get_nodes_by_op_type("StreamingDataflowPartition")
    sdp = [n for n in sdp_nodes if n.name == f"GenericPartition_{PART_IDX}"][0]

    with open(FOLDING_JSON) as f:
        folding_block = json.load(f)
    per_layer = folding_block["per_layer"]
    extra_nodes = folding_block.get("extra_nodes")
    inter_block_fifos = folding_block.get("inter_block_fifos")
    intra_block_fifos = folding_block.get("intra_block_fifos")
    logical = load_partition_logical_names(PREAMBLE_DIR, CONV_ORDER)

    print("=" * 80, flush=True)
    print("STARTING build_partition_folding_config", flush=True)
    print("=" * 80, flush=True)

    fc, fifo_plan = build_partition_folding_config(
        getCustomOp(sdp).get_nodeattr("model"), sdp.name, logical[PART_IDX][0], per_layer, cfg,
        tag=f"p{PART_IDX}", extra_nodes=extra_nodes, conv_order_file=CONV_ORDER,
        inter_block_fifos=inter_block_fifos, intra_block_fifos=intra_block_fifos,
    )

    role_of_node = fifo_plan.get("role_of_node", {})
    print("=" * 80, flush=True)
    print("down2-related final role_of_node entries:", flush=True)
    for node_name, role in role_of_node.items():
        if role.startswith("down2."):
            print(f"  {node_name} -> {role}", flush=True)


if __name__ == "__main__":
    main()
