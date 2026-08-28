"""One-off diagnostic: dump every node in [down1_start, down2_start) --
i.e. the "partition 1" (down1+regular1) index range -- in graph order,
showing op_type + partition_id nodeattr (None for non-HW/unassigned), to
find which non-HW leftover node is sandwiched between two partition-1 HW
nodes and causing the reconvergence cycle. Not part of the pipeline --
run manually, delete after use.
"""
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

sys.path.insert(0, ".")
from finn_stage_partition import find_stage_boundaries  # noqa: E402

MODEL_FILE = sys.argv[1]
model = ModelWrapper(MODEL_FILE)

down1_start, down2_start, up4_start, up5_start = find_stage_boundaries(model)
print(f"down1_start={down1_start} down2_start={down2_start} up4_start={up4_start} up5_start={up5_start}")

all_nodes = list(model.graph.node)
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
hi = int(sys.argv[3]) if len(sys.argv) > 3 else down1_start
for idx in range(lo, hi):
    node = all_nodes[idx]
    try:
        pid = getCustomOp(node).get_nodeattr("partition_id")
    except Exception as e:
        pid = f"ERR({e.__class__.__name__})"
    ins = list(node.input)
    outs = list(node.output)
    print(f"[{idx}] {node.op_type:20s} name={node.name:30s} pid={pid} in={ins} out={outs}")
