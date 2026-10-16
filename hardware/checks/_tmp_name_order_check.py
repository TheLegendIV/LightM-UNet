"""Do the post-hoc node names of a partition stay aligned with the names the folding bridge saw?

Stage A: names the bridge/apply_folding see (specialize -> GiveUniqueNodeNames -> target_fps -> GiveUniqueNodeNames).
Stage B: after the per-partition prefix rename (GiveUniqueNodeNames(prefix)).
Stage C: after step_set_fifo_depths_fixed2 (InsertDWC, InsertFIFO, SpecializeLayers, GiveUniqueNodeNames) -- where
         step_force_fifo_depths_from_milp applies the bridge's role_of_node.
Prints the Thresholding nodes in list order with their channel count at every stage.
Usage (container): python3 _tmp_name_order_check.py <preamble_dir> <partition idx>"""
import dataclasses
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
preamble_dir, idx, folding_file = sys.argv[1], int(sys.argv[2]), sys.argv[3]
_real = sys.argv
sys.argv = sys.argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402
from finn.builder.build_dataflow_steps import step_specialize_layers, step_target_fps_parallelization, step_apply_folding_config  # noqa: E402
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC  # noqa: E402
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO  # noqa: E402
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers  # noqa: E402
from finn_partition_build_steps import step_create_dataflow_partition_multi  # noqa: E402
import finn_s12_build_steps as bs  # noqa: E402

bs.install_relaxed_stage_boundaries()
out = "/tmp/_nameorder"
os.makedirs(out, exist_ok=True)
cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=out, folding_config_file=folding_file)
parent = step_create_dataflow_partition_multi(ModelWrapper(f"{preamble_dir}/intermediate_models/assign_stage_partition_ids_8way.onnx"), cfg)
sdp = parent.get_nodes_by_op_type("StreamingDataflowPartition")[idx]
prefix = sdp.name + "_"
m = ModelWrapper(getCustomOp(sdp).get_nodeattr("model"))


def thr(m):
    rows = []
    for n in m.graph.node:
        if n.op_type.startswith("Thresholding"):
            prod = m.find_producer(n.input[0])
            rows.append((n.name, getCustomOp(n).get_nodeattr("NumChannels"), prod.op_type if prod is not None else None))
    return rows


m = step_specialize_layers(m, cfg)
m = m.transform(GiveUniqueNodeNames())
m = m.transform(GiveReadableTensorNames())
m = step_target_fps_parallelization(m, cfg)
m = step_apply_folding_config(m, cfg)
print("A (bridge names):", thr(m)[:6])
m = m.transform(GiveUniqueNodeNames(prefix))
print("B (prefix names):", [(n.replace(prefix, ""), c, p) for n, c, p in thr(m)[:6]])
m = m.transform(InsertDWC())
m = m.transform(InsertFIFO(create_shallow_fifos=True, vivado_ram_style=cfg.large_fifo_mem_style))
m = m.transform(SpecializeLayers(cfg._resolve_fpga_part()))
print("C0 (before rename, prefix names):", [(n.replace(prefix, ""), c, p) for n, c, p in thr(m)[:6]])
m = m.transform(GiveUniqueNodeNames())
print("C (after rename):", thr(m)[:6])
