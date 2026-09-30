"""Analytical cycle/throughput estimate for the autofold PE/SIMD choice,
computable right now without waiting for OOC synth (uses AnnotateCycles +
dataflow_performance -- the same analysis FINN's own step_generate_estimate_
reports would run, just not included in this custom step chain)."""
import sys
import dataclasses

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles  # noqa: E402
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance  # noqa: E402

CKPT = sys.argv[1]
cfg = base.cfg_stitched_ip_partitioned_8way
clk_ns = cfg.synth_clk_period_ns

model = ModelWrapper(CKPT)
model = model.transform(AnnotateCycles())
perf = model.analysis(dataflow_performance)

max_cycles = perf["max_cycles"]
max_cycles_node = perf["max_cycles_node_name"]
est_fps = 1.0 / (max_cycles * clk_ns * 1e-9)

print(f"synth_clk_period_ns : {clk_ns}")
print(f"max_cycles          : {max_cycles}  (bottleneck node: {max_cycles_node})")
print(f"estimated throughput: {est_fps:.2f} FPS  (target was 250 FPS)")
print(f"target met (analytically, assuming clk achieved): {est_fps >= 250}")
