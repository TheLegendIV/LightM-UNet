"""Cheap check (no partitioning/ipgen/rtlsim): does step_compose_consecutive_thresholds actually
collapse residual_add->out_act (and skip_quant->residual_add? no -- just the Add->Thr(residual_add)
->Thr(out_act) chain) on the REAL flat S12-dense graph, right before step_enet_convert_to_hw runs?

Loads the preamble's own saved `_fixup_degenerate_signed_bias.onnx` checkpoint (the last QONNX-level
checkpoint before HW conversion, still plain MultiThreshold nodes) -- same input step_enet_convert_to_hw
itself consumed for the REAL build we've been probing. Runs the compose pass, reports the MultiThreshold
count before/after, then runs step_enet_convert_to_hw on the composed graph and confirms each block's
post-Add chain now lowers to exactly ONE Thresholding node (not two).
"""
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

PREAMBLE_DIR = ("/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
                "S12_dense_256_u4_analytical_v1_finn_calibrated_rtl_mvau_256x256_preamble_20261005_031021")
PROBE_OUT = "/tmp/_tmp_probe_compose_thresholds_out"
os.makedirs(PROBE_OUT, exist_ok=True)
os.environ["FINN_BUILD_DIR"] = os.path.join(PROBE_OUT, "finn_build_tmp")
os.makedirs(os.environ["FINN_BUILD_DIR"], exist_ok=True)

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402

from finn_compose_thresholds import step_compose_consecutive_thresholds  # noqa: E402
from finn_enet_build_fixups import step_enet_convert_to_hw  # noqa: E402

import dataclasses
_real_argv, sys.argv = sys.argv, sys.argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=PROBE_OUT)

ckpt = os.path.join(PREAMBLE_DIR, "intermediate_models", "_fixup_degenerate_signed_bias.onnx")
m = ModelWrapper(ckpt)

before = sum(n.op_type == "MultiThreshold" for n in m.graph.node)
print(f"MultiThreshold nodes BEFORE compose: {before}")

m = step_compose_consecutive_thresholds(m, cfg)
m = m.transform(GiveUniqueNodeNames())
m = m.transform(GiveReadableTensorNames())

after = sum(n.op_type == "MultiThreshold" for n in m.graph.node)
print(f"MultiThreshold nodes AFTER compose:  {after}  (removed {before - after})")
m.save(os.path.join(PROBE_OUT, "composed_pre_convert.onnx"))

# convert to hw on the composed graph, confirm every Add is followed by exactly one Thresholding
m2 = step_enet_convert_to_hw(m, cfg)
m2 = m2.transform(GiveUniqueNodeNames())
m2 = m2.transform(GiveReadableTensorNames())
m2.save(os.path.join(PROBE_OUT, "composed_after_convert_to_hw.onnx"))

thr_total = sum(n.op_type.startswith("Thresholding") for n in m2.graph.node)
add_total = sum(n.op_type.startswith("AddStreams") for n in m2.graph.node)
print(f"\nAfter convert_to_hw: {add_total} AddStreams nodes, {thr_total} Thresholding nodes total")

bad = []
for n in m2.graph.node:
    if not n.op_type.startswith("AddStreams"):
        continue
    out_t = n.output[0]
    consumers = m2.find_consumers(out_t)
    n_thr_chain = 0
    cur_tensor = out_t
    while True:
        cons = m2.find_consumers(cur_tensor)
        if not cons or len(cons) != 1 or not cons[0].op_type.startswith("Thresholding"):
            break
        n_thr_chain += 1
        cur_tensor = cons[0].output[0]
    status = "OK (1 threshold)" if n_thr_chain == 1 else f"MISMATCH ({n_thr_chain} thresholds chained)"
    print(f"  {n.name:30s} -> {n_thr_chain} Thresholding node(s) after it: {status}")
    if n_thr_chain != 1:
        bad.append(n.name)

print(f"\n{len(bad)} AddStreams node(s) NOT followed by exactly one Thresholding: {bad}")
print("RESULT:", "PASS -- merge collapsed every residual join to one threshold" if not bad else "FAIL -- see mismatches above")
