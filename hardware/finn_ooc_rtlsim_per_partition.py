"""Fallback to the "OOC each partition separately" plan: instead of
combining all 8 partitions' Verilog into one flat design for Vivado OOC
synthesis (which kept hitting new bugs in that merge/flatten pipeline --
basename collisions, zsh NOMATCH aborts, SV package compile-order), just
run FINN's own standard (unmodified, proven) single-model
step_out_of_context_synthesis and step_measure_rtlsim_performance
directly on each partition's own already-built stitched IP. Each
partition's kernel model already has valid "vivado_stitch_proj" +
"wrapper_filename" metadata from its own CreateStitchedIP run in
step_build_all_partitions, so this needs none of the combine-step
machinery -- just load and run.

Usage (inside the FINN container):
    python3 finn_ooc_rtlsim_per_partition.py <partition_idx 0-7>

Writes reports to
finn_deployment_outputs/<run>/per_partition_reports/<partition_name>/report/
"""

import os
import sys
import dataclasses

if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(1)

partition_idx = int(sys.argv[1])
sys.argv = [sys.argv[0]]  # finn_enet_ip_build_partitioned_8way parses sys.argv itself

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_out_of_context_synthesis,
    step_measure_rtlsim_performance,
)

# base.OUTPUT_DIR is regenerated with datetime.now() at import time -- it
# never matches the already-completed run we actually want to read from.
# Point at that specific existing run's directory instead.
RUN_DIR = os.path.join(
    base.ENET_DIR,
    "finn_deployment_outputs",
    "stitched_ip_partitioned8way_quantEnet_s19_double_mid_int8_largefifo_rtlsim_20260820_101224",
)
PARENT_CKPT = os.path.join(RUN_DIR, "intermediate_models", "step_build_all_partitions_capped.onnx")

if __name__ == "__main__":
    parent = ModelWrapper(PARENT_CKPT)
    sdp_nodes = parent.get_nodes_by_op_type("StreamingDataflowPartition")
    node = sdp_nodes[partition_idx]
    sdp_inst = getCustomOp(node)
    model_fn = sdp_inst.get_nodeattr("model")
    model = ModelWrapper(model_fn)

    out_dir = os.path.join(RUN_DIR, "per_partition_reports", node.name)
    os.makedirs(out_dir, exist_ok=True)

    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=out_dir)

    print("=== [%s] OOC synthesis ===" % node.name)
    model = step_out_of_context_synthesis(model, cfg)

    print("=== [%s] rtlsim performance ===" % node.name)
    model = step_measure_rtlsim_performance(model, cfg)

    print("=== [%s] done, reports in %s/report ===" % (node.name, out_dir))
