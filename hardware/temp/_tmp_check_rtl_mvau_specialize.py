"""Throwaway check: after the rtl_mvau preamble, does step_specialize_layers
actually lower every MVAU/VVAU/Thresholding node to the _rtl backend (not
_hls)? Run BEFORE launching the expensive 8-way stitched-IP+OOC build."""
import sys
import dataclasses
from collections import Counter

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from finn.builder.build_dataflow_steps import step_specialize_layers  # noqa: E402

preamble_dir = sys.argv[1]
ckpt = f"{preamble_dir}/intermediate_models/step_enet_convert_to_hw_rtl_mvau.onnx"
print(f"Loading {ckpt}")
model = ModelWrapper(ckpt)

before = Counter(n.op_type for n in model.graph.node)
print("Before specialize:", dict(before))

cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir="/tmp/_rtl_mvau_specialize_check")
model = step_specialize_layers(model, cfg)

after = Counter(n.op_type for n in model.graph.node)
print("After specialize:", dict(after))

hls_offenders = [n for n in model.graph.node if n.op_type in ("MVAU_hls", "VVAU_hls", "Thresholding_hls")]
print(f"\n{len(hls_offenders)} node(s) landed on _hls backend (expected: 0):")
for n in hls_offenders:
    inst = getCustomOp(n)
    extra = {}
    for attr in ("noActivation", "weightDataType", "inputDataType"):
        try:
            extra[attr] = inst.get_nodeattr(attr)
        except Exception:
            pass
    print(f"  {n.name:30s} {n.op_type:16s} {extra}")

n_thresh_standalone = sum(1 for n in model.graph.node if n.op_type in ("Thresholding_hls", "Thresholding_rtl"))
print(f"\nstandalone Thresholding nodes (any backend): {n_thresh_standalone}")
print("PASS" if len(hls_offenders) == 0 else "FAIL")
