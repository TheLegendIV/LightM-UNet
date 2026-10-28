"""Read-only debug: inspect partition 1's raw (pre-specialize) graph around AddStreams for down1/down2
to see why _find_join_thresholds fails to find the skip_quant/thr_s Thresholding node."""
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
os.environ.setdefault("FINN_ROOT", "/home/thelegendiv/finn")
os.environ["FINN_BUILD_DIR"] = "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/debug_p1_role_scratch"

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn_enet_ip_build_partitioned_8way as base
from finn_partition_build_steps import step_create_dataflow_partition_multi
from finn_s12_build_steps import load_partition_logical_names, WEIGHT_OP_TYPES, THRESH_OP_TYPES

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
PREAMBLE_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", "S12_dense_256_u4_analytical_v1_ft15ep_preamble_20261005_220558")
CONV_ORDER = os.path.join(ENET_DIR, "quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_conv_order.json")
PART_IDX = 1

cfg = base.cfg_stitched_ip_partitioned_8way
flat_ckpt = os.path.join(PREAMBLE_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
parent = step_create_dataflow_partition_multi(ModelWrapper(flat_ckpt), cfg)
sdp_nodes = parent.get_nodes_by_op_type("StreamingDataflowPartition")
sdp = [n for n in sdp_nodes if n.name == f"GenericPartition_{PART_IDX}"][0]
kernel_model = ModelWrapper(getCustomOp(sdp).get_nodeattr("model"))

adds = [n for n in kernel_model.graph.node if n.op_type.startswith("AddStreams")]
print("num AddStreams:", len(adds))
for i, add in enumerate(adds[:2]):
    print(f"--- add[{i}] = {add.name} ---")
    for inp in add.input:
        prod = kernel_model.find_producer(inp)
        print("  input producer:", prod.name if prod else None, prod.op_type if prod else None)
        if prod is not None:
            prod_in = kernel_model.find_producer(prod.input[0])
            print("    its producer:", prod_in.name if prod_in else None, prod_in.op_type if prod_in else None)

logical = load_partition_logical_names(PREAMBLE_DIR, CONV_ORDER)
print("logical names for partition 1 (first 20):", logical[PART_IDX][0][:20])
