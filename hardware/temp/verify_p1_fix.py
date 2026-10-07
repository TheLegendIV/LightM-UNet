import dataclasses
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
os.environ.setdefault("FINN_ROOT", "/home/thelegendiv/finn")

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
BUILD_TAG = "debug_p1_verify_fix_v1"
os.environ["FINN_BUILD_DIR"] = os.path.join(ENET_DIR, "finn_build_tmp", BUILD_TAG)

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn_enet_ip_build_partitioned_8way as base
from finn_partition_build_steps import step_create_dataflow_partition_multi
from finn_s12_build_steps import build_partition_folding_config, load_partition_logical_names

PREAMBLE_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", "S12_dense_256_u4_analytical_v1_ft15ep_preamble_20261005_220558")
CONV_ORDER = os.path.join(ENET_DIR, "quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_conv_order.json")
FOLDING_JSON = os.path.join(ENET_DIR, "layer_bits_folding_S12_dense_256_u4_analytical_v1_int6_fps250_lat200_20261006.json")
PART_IDX = 1
OUTPUT_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", f"debug_{BUILD_TAG}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(FOLDING_JSON) as f:
    folding_block = json.load(f)
per_layer = folding_block["per_layer"]
extra_nodes = folding_block.get("extra_nodes")
inter_block_fifos = folding_block.get("inter_block_fifos")
intra_block_fifos = folding_block.get("intra_block_fifos")

cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)
flat_ckpt = os.path.join(PREAMBLE_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
parent = step_create_dataflow_partition_multi(ModelWrapper(flat_ckpt), cfg)
sdp_nodes = parent.get_nodes_by_op_type("StreamingDataflowPartition")
sdp = [n for n in sdp_nodes if n.name == f"GenericPartition_{PART_IDX}"][0]
logical = load_partition_logical_names(PREAMBLE_DIR, CONV_ORDER)

fc, fifo_plan = build_partition_folding_config(
    getCustomOp(sdp).get_nodeattr("model"), sdp.name, logical[PART_IDX][0], per_layer, cfg,
    tag=f"p{PART_IDX}", extra_nodes=extra_nodes, conv_order_file=CONV_ORDER,
    inter_block_fifos=inter_block_fifos, intra_block_fifos=intra_block_fifos,
)
role_of_node = fifo_plan["role_of_node"]
node_of_role = {v: k for k, v in role_of_node.items()}
for b in ("down1", "down2"):
    for suffix in ("dup", "swg_r", "maxpool", "mvau_s", "thr_s", "add"):
        print(f"{b}.{suffix} ->", node_of_role.get(f"{b}.{suffix}"))

wanted = fifo_plan["wanted"]
for key in ("down1.thr_s=>down1.add", "down2.thr_s=>down2.add", "down1.dup=>down1.swg_r", "down1.swg_r=>down1.mvau_r"):
    print(key, "wanted:", wanted.get(key))
