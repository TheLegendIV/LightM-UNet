"""Standalone validation: run step_force_fifo_depths_from_milp against a real, already-FIFO-inserted
partition 0 graph (pre-autosize, topology is unaffected) using the fix2 fifo_plan, and print the full
per-FIFO match report so we can confirm real match counts/roles without a multi-hour full build."""
import json
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402

from finn_s12_build_steps import step_force_fifo_depths_from_milp  # noqa: E402

MODEL = ("/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
         "S12_dense_256_u4_analytical_v1_milpfold_8way_20261005_135935/intermediate_models/"
         "supported_op_partitions/partition_0_prefifo_autosize.onnx")
PLAN = ("/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
        "S12_dense_256_u4_analytical_v1_validate_fifo_fix2_milpfold_partition0_20261005_155151/"
        "fifo_plan_partition0.json")

model = ModelWrapper(MODEL)
fifo_plan = json.load(open(PLAN))
_, report = step_force_fifo_depths_from_milp(model, fifo_plan)

print(f"\n{'fifo':40s} {'producer_role':22s} {'consumer_role':22s} {'stock':>6s} {'forced':>6s}  notes")
for e in report:
    notes = []
    if e.get("forced_via_virtual_dwc_collapse"):
        notes.append("virtual_dwc")
    if e.get("forced_via_dwc_bridge"):
        notes.append("dwc_bridge")
    if e.get("forced_via_producer_only_inter_block"):
        notes.append("producer_only")
    print(f"{e['fifo']:40s} {str(e['producer_role']):22s} {str(e['consumer_role']):22s} "
          f"{str(e['stock_depth']):>6s} {str(e['forced_depth']):>6s}  {','.join(notes)}")
