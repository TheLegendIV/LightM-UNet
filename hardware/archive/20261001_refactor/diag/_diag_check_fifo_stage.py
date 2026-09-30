import sys
import json
import dataclasses

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

import finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_8way_full as m
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from finn.builder.build_dataflow_steps import (
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
    step_hw_codegen,
    step_hw_ipgen,
)
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

OUTDIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_8way_full_20260913_095107"
BASE = f"{OUTDIR}/intermediate_models/supported_op_partitions"

cfg = dataclasses.replace(m.base.cfg_stitched_ip_partitioned_8way, output_dir=OUTDIR)

i = 0
fn = f"{BASE}/partition_{i}.onnx"
ffile = f"{OUTDIR}/hawq_folding_config_partition{i}.json"
prefix = f"StreamingDataflowPartition_{i}_"

part_cfg = dataclasses.replace(cfg, folding_config_file=ffile)
kernel_model = ModelWrapper(fn)
kernel_model = step_specialize_layers(kernel_model, part_cfg)
kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
kernel_model = kernel_model.transform(GiveReadableTensorNames())
kernel_model = step_target_fps_parallelization(kernel_model, part_cfg)
kernel_model = step_apply_folding_config(kernel_model, part_cfg)
kernel_model = step_minimize_bit_width(kernel_model, part_cfg)
kernel_model = m.step_fix_weight_dtype_bipolar_bug(kernel_model, part_cfg)
kernel_model = m.step_force_dsp(kernel_model, part_cfg)
kernel_model = step_hw_codegen(kernel_model, part_cfg)
kernel_model = step_hw_ipgen(kernel_model, part_cfg)

print("=== before FIFO insertion ===")
print("graph.output:", [o.name for o in kernel_model.graph.output])
for o in kernel_model.graph.output:
    p = kernel_model.find_producer(o.name)
    print(" ", o.name, "producer=", p.name if p is not None else None)

kernel_model = kernel_model.transform(InsertDWC())
kernel_model = kernel_model.transform(InsertFIFO(create_shallow_fifos=True))
kernel_model = kernel_model.transform(SpecializeLayers(part_cfg._resolve_fpga_part()))
kernel_model = kernel_model.transform(GiveUniqueNodeNames())
kernel_model = kernel_model.transform(GiveReadableTensorNames())

print("=== after FIFO insertion + rename ===")
print("graph.output:", [o.name for o in kernel_model.graph.output])
for o in kernel_model.graph.output:
    p = kernel_model.find_producer(o.name)
    print(" ", o.name, "producer=", p.name if p is not None else None)

print("=== all node names + outputs ===")
for node in kernel_model.graph.node:
    print(" ", node.op_type, node.name, "-> outputs:", list(node.output))
