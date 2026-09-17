"""Replay the exact per-partition build pipeline (up to step_minimize_bit_width)
on partition 1 alone, catching MinimizeAccumulatorWidth's per-node assert to
pinpoint which Thresholding node has an inputDataType/threshold-sign mismatch."""
import sys
import dataclasses
import numpy as np
sys.argv = [sys.argv[0]]  # avoid argv pollution from base module import

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames

import finn_enet_ip_build_partitioned_8way as base
from finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full import (
    step_specialize_layers, step_target_fps_parallelization, step_apply_folding_config,
)

OUTDIR = "finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_20260917_011115"
PARTITION_MODEL = f"{OUTDIR}/intermediate_models/supported_op_partitions/partition_1.onnx"
FOLDING_CONFIG = f"{OUTDIR}/hawq_folding_config_partition1.json"

cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTDIR,
                          folding_config_file=FOLDING_CONFIG)

model = ModelWrapper(PARTITION_MODEL)
model = step_specialize_layers(model, cfg)
model = model.transform(GiveUniqueNodeNames("partition1_"))
model = model.transform(GiveReadableTensorNames())
model = step_target_fps_parallelization(model, cfg)
model = step_apply_folding_config(model, cfg)

print("Replayed pipeline up to (not including) step_minimize_bit_width. Checking each Thresholding node...")
for n in model.graph.node:
    if not n.op_type.startswith("Thresholding"):
        continue
    inst = getCustomOp(n)
    in_dt = inst.get_input_datatype()
    thresholds = model.get_initializer(n.input[1])
    has_neg = (thresholds < 0).any()
    status = "OK"
    if (not in_dt.signed()) and has_neg:
        status = "*** WOULD CRASH (unsigned inputDataType, negative thresholds) ***"
    print(f"{n.name}: op_type={n.op_type} inputDataType={in_dt} thr_min={thresholds.min()} "
          f"thr_max={thresholds.max()} producer={model.find_producer(n.input[0]).name if model.find_producer(n.input[0]) else None} -- {status}")

print("\nNow replaying MinimizeAccumulatorWidth's own node-by-node loop (with InferDataTypes "
      "cascading after each node, exactly like the real transform) to find the exact crash point...")
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.util.fpgadataflow import is_fpgadataflow_node
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth  # noqa: F401

for node_id in range(len(model.graph.node)):
    n = model.graph.node[node_id]
    if not is_fpgadataflow_node(n):
        continue
    inst = getCustomOp(n)
    if not hasattr(inst, "minimize_accumulator_width"):
        continue
    try:
        inst.minimize_accumulator_width(model)
    except AssertionError as e:
        in_dt = inst.get_input_datatype()
        thresholds = model.get_initializer(n.input[1]) if len(n.input) > 1 else None
        prod = model.find_producer(n.input[0])
        print(f"\n*** CRASH at {n.name} (op_type={n.op_type}) ***")
        print(f"  inputDataType={in_dt} (signed={in_dt.signed()})")
        if thresholds is not None:
            print(f"  threshold min={thresholds.min()} max={thresholds.max()}")
        print(f"  producer={prod.name if prod else None} (op_type={prod.op_type if prod else None})")
        if prod is not None:
            prod_inst = getCustomOp(prod)
            try:
                print(f"  producer output dtype={model.get_tensor_datatype(n.input[0])}")
            except Exception:
                pass
        print(f"  error: {e}")
        break
    model = model.transform(InferDataTypes())
else:
    print("No crash reproduced -- all nodes processed OK.")

