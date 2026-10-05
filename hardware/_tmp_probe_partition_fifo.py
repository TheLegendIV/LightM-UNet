"""Standalone probe: run ONE partition through the real pipeline (specialize -> fold -> codegen ->
ipgen) far enough for folded shapes to settle, then InsertFIFO(create_shallow_fifos=True) directly
(skipping step_set_fifo_depths' expensive rtlsim depth-search -- InsertAndSetFIFODepths.apply() calls
this exact same InsertFIFO first, then only RESIZES existing FIFOs via rtlsim, never adds/removes
placement, so topology is identical either way). Then call step_force_fifo_depths_from_milp and print
the match report. NOTE: skipping codegen/ipgen (tried first) breaks InsertFIFO's folded-shape
assertion -- several op types (FMPadding_rtl, ConvolutionInputGenerator_rtl, AddStreams_hls,
ChannelwiseOp_hls, DuplicateStreams_hls) get no entry in the MILP-bridged folding config (it only
covers MILP-modeled roles) and SetFolding can't auto-fold some of them either; something in
codegen/ipgen normalizes their folded shapes before FIFO insertion -- so codegen/ipgen stay, only the
rtlsim measurement is skipped. This only validates role/producer/consumer MATCHING, not depth
correctness (every FIFO here starts at the shallow placeholder depth of 2).

Usage: python3 _tmp_probe_partition_fifo.py <partition_idx>
Reuses the already-saved hawq_folding_config_partition<idx>.json / fifo_plan_partition<idx>.json
from the prior --bridge-only --partitions all run (BRIDGE_DIR below) -- no need to re-derive them.
"""
import dataclasses
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

PREAMBLE_DIR = ("/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
                "S12_dense_256_u4_analytical_v1_finn_calibrated_rtl_mvau_256x256_composed_preamble_20261005_164026")
BRIDGE_DIR = ("/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
              "S12_dense_256_u4_analytical_v1_finn_calibrated_composed_milpfold_partition0_1_6_20261005_165320")
PROBE_OUT = "/tmp/_tmp_probe_partition_fifo_out"
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

from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_specialize_layers, step_target_fps_parallelization, step_apply_folding_config,
    step_hw_codegen, step_hw_ipgen,
)
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO  # noqa: E402
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC  # noqa: E402
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers  # noqa: E402
from finn_partition_build_steps import step_create_dataflow_partition_multi  # noqa: E402
from finn_s12_build_steps import (  # noqa: E402
    install_relaxed_stage_boundaries, step_fix_weight_dtype_bipolar_bug, step_force_dsp,
    step_force_fifo_depths_from_milp, step_minimize_bit_width_standalone_thresh_aware,
)

install_relaxed_stage_boundaries()

cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=PROBE_OUT)

flat_ckpt = os.path.join(PREAMBLE_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
parent = step_create_dataflow_partition_multi(ModelWrapper(flat_ckpt), cfg)
sdp_nodes = parent.get_nodes_by_op_type("StreamingDataflowPartition")
assert len(sdp_nodes) == 8, len(sdp_nodes)
partition_model_fn = getCustomOp(sdp_nodes[idx]).get_nodeattr("model")

folding_file = os.path.join(BRIDGE_DIR, f"hawq_folding_config_partition{idx}.json")
fifo_plan = json.load(open(os.path.join(BRIDGE_DIR, f"fifo_plan_partition{idx}.json")))
cfg = dataclasses.replace(cfg, folding_config_file=folding_file)

m = ModelWrapper(partition_model_fn)
m = step_specialize_layers(m, cfg)
m = m.transform(GiveUniqueNodeNames())
m = m.transform(GiveReadableTensorNames())
m = step_target_fps_parallelization(m, cfg)
m = step_apply_folding_config(m, cfg)
m = step_minimize_bit_width_standalone_thresh_aware(m, cfg)
m = step_fix_weight_dtype_bipolar_bug(m, cfg)
m = step_force_dsp(m, cfg)
m = step_hw_codegen(m, cfg)
m = step_hw_ipgen(m, cfg)
m = m.transform(InsertDWC())
m = m.transform(InsertFIFO(create_shallow_fifos=True))
m = m.transform(SpecializeLayers(cfg._resolve_fpga_part()))
m = m.transform(GiveUniqueNodeNames())
m = m.transform(GiveReadableTensorNames())
m.save(os.path.join(PROBE_OUT, f"partition_{idx}_shallow_fifos.onnx"))

_, report = step_force_fifo_depths_from_milp(m, fifo_plan)
print(f"\n{'fifo':40s} {'producer_role':24s} {'consumer_role':24s} {'stock':>6s} {'forced':>6s}  notes")
for e in report:
    notes = []
    if e.get("forced_via_virtual_dwc_collapse"):
        notes.append("virtual_dwc")
    if e.get("forced_via_dwc_bridge"):
        notes.append("dwc_bridge")
    if e.get("forced_via_producer_only_inter_block"):
        notes.append("producer_only")
    print(f"{e['fifo']:40s} {str(e['producer_role']):24s} {str(e['consumer_role']):24s} "
          f"{str(e['stock_depth']):>6s} {str(e['forced_depth']):>6s}  {','.join(notes)}")
