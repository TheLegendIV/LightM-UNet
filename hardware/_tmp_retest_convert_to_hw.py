"""One-off validation: re-run ONLY step_enet_convert_to_hw against the
saved _fixup_degenerate_signed_bias.onnx checkpoint (the input to that
step) to quickly validate the MakeMaxPoolNHWC + MoveTransposePastJoinConcat
fix without re-running the whole ~25min preamble. Not part of the
pipeline -- run manually, delete after use.
"""
import sys

sys.path.insert(0, ".")
from qonnx.core.modelwrapper import ModelWrapper
from finn_enet_build import step_enet_convert_to_hw
from finn_stage_partition import assign_stage_partition_ids_8way, find_stage_boundaries

MODEL_FILE = sys.argv[1]
model = ModelWrapper(MODEL_FILE)
model = step_enet_convert_to_hw(model, None)
model = assign_stage_partition_ids_8way(model, None)
out_file = sys.argv[2] if len(sys.argv) > 2 else "/tmp/_tmp_convert_to_hw_retest.onnx"
model.save(out_file)
print("saved:", out_file)

db = find_stage_boundaries(model)
print("boundaries:", db)
