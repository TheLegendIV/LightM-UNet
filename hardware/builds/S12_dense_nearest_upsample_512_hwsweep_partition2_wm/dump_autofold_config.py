"""Extract the auto-fold PE/SIMD folding config chosen by FINN's own
step_target_fps_parallelization for the baseline_both_off_autofold partition-2
build (no MILP bridge was applied to this build -- step_apply_folding_config
was skipped entirely, see finn_ooc_partition2_trained.py). PE/SIMD values are
set before step_minimize_bit_width/step_hw_codegen/step_hw_ipgen/
step_set_fifo_depths, none of which touch folding -- so the prefifo_autosize
checkpoint still holds the exact auto-fold result, no need to wait for
stitching/OOC synth to finish.

Output schema matches the MILP-bridged hawq_folding_config_partition2.json
files (dict of node_name -> {"PE": int, "SIMD": int}) for direct comparison.

Usage: python3 dump_autofold_config.py <prefifo_autosize_checkpoint.onnx> <output.json>
"""
import sys
import json

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from finn.util.fpgadataflow import is_fpgadataflow_node  # noqa: E402

CKPT = sys.argv[1]
OUT_JSON = sys.argv[2]

model = ModelWrapper(CKPT)
config = {"Defaults": {}}
for node in model.graph.node:
    if not is_fpgadataflow_node(node):
        continue
    inst = getCustomOp(node)
    entry = {}
    for attr in ("PE", "SIMD"):
        try:
            entry[attr] = inst.get_nodeattr(attr)
        except Exception:
            pass
    if entry:
        config[node.name] = entry

with open(OUT_JSON, "w") as f:
    json.dump(config, f, indent=2)
print(f"Wrote {len(config) - 1} node folding entries to {OUT_JSON}")
