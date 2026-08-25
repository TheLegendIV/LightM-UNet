"""Direct node-for-node diff of every Thresholding_rtl node's memory-relevant
attributes (NumChannels, PE, numSteps/weightDataType-derived depth) between
the leaky and ReLU-only partition-2 variants -- res_estimation() reports 0
BRAM for RTL thresholding backends (it doesn't model RTL memory geometry),
so this inspects the actual nodeattrs that drive real BRAM depth instead
(see thresholding_rtl.py's get_pe_mem_geometries(): depth = cf * 2**x for x
in 0..(odt_bits-1), cf = NumChannels/PE).

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \
        /home/thelegendiv/finn/notebooks/enet/finn_relu_vs_leaky_threshold_diff.py
"""
import os
import sys
import dataclasses

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402

import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
from finn_partition_build_steps import step_create_dataflow_partition_multi  # noqa: E402
from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
)

PARTITION_IDX = 2
VARIANTS = {
    "leaky": os.path.join(
        base.ENET_DIR, "finn_deployment_outputs",
        "stitched_ip_partitioned8way_quantEnet_s19_double_mid_int8_largefifo_rtlsim_20260820_101224",
        "intermediate_models", "assign_stage_partition_ids_8way.onnx",
    ),
    "relu-only": os.path.join(
        base.ENET_DIR, "finn_deployment_outputs", "relu_only_preamble_20260824_195233",
        "intermediate_models", "assign_stage_partition_ids_8way.onnx",
    ),
}


def build_partition2(source_ckpt, cfg):
    flat_model = ModelWrapper(source_ckpt)
    parent_model = step_create_dataflow_partition_multi(flat_model, cfg)
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[PARTITION_IDX]
    prefix = sdp_node.name + "_"
    partition_model_fn = getCustomOp(sdp_node).get_nodeattr("model")
    kernel_model = ModelWrapper(partition_model_fn)
    kernel_model = step_specialize_layers(kernel_model, cfg)
    kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
    kernel_model = kernel_model.transform(GiveReadableTensorNames())
    kernel_model = step_target_fps_parallelization(kernel_model, cfg)
    kernel_model = step_apply_folding_config(kernel_model, cfg)
    kernel_model = step_minimize_bit_width(kernel_model, cfg)
    return kernel_model


def threshold_geometries(kernel_model):
    out = []
    for node in kernel_model.graph.node:
        if node.op_type != "Thresholding_rtl":
            continue
        inst = getCustomOp(node)
        nc = inst.get_nodeattr("NumChannels")
        pe = inst.get_nodeattr("PE")
        odt = inst.get_nodeattr("outputDataType")
        wdt = inst.get_nodeattr("weightDataType")
        cf = nc // pe
        out.append((node.name, nc, pe, cf, odt, wdt))
    return out


if __name__ == "__main__":
    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir="/tmp/_thresh_diff_scratch")
    geoms = {}
    for label, ckpt in VARIANTS.items():
        km = build_partition2(ckpt, cfg)
        geoms[label] = threshold_geometries(km)
        print(f"{label}: {len(geoms[label])} Thresholding_rtl nodes")

    a, b = geoms["leaky"], geoms["relu-only"]
    identical = (len(a) == len(b)) and all(
        (x[1], x[2], x[3], x[4], x[5]) == (y[1], y[2], y[3], y[4], y[5]) for x, y in zip(a, b)
    )
    print(f"\nGeometry (NumChannels, PE, cf=NumChannels/PE, outputDT, weightDT) identical node-for-node: {identical}")
    if not identical:
        for i, (x, y) in enumerate(zip(a, b)):
            if (x[1], x[2], x[3], x[4], x[5]) != (y[1], y[2], y[3], y[4], y[5]):
                print(f"  DIFF at index {i}: leaky={x}  relu={y}")
    else:
        print("Sample (first 5):")
        for x in a[:5]:
            print(f"  {x}")
