"""One-off: build partition 5 ONLY through FIFO auto-sizing (step_hw_codegen ->
step_hw_ipgen -> step_set_fifo_depths), using the corrected (VVAU-PE-scaled-up)
folding already baked into the real partition_5.onnx by
finn_refix_p5p6_folding_v4_512x512.py, to check whether the fix reduces
autosized FIFO depths vs the original crashed build. Stops right after
step_set_fifo_depths and saves that checkpoint -- does NOT run SplitLargeFIFOs /
PrepareIP / HLSSynthIP / CreateStitchedIP / SynthOutOfContext, and does NOT
touch partition 6 or any other partition.

Uses a FRESH build dir (GenericPartition_5_refix), not the original
GenericPartition_5 (which holds stale HLS/IP artifacts from the crashed,
pre-fix folding -- see repo memory finn_gotchas.md's stale-IP-cache-reuse
entry).
"""
import dataclasses
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_ooc_12_dense_relu_warmstart150ep_alpha05_trained_rtl_mvau_8way_full_v4_512x512 as buildmod  # noqa: E402
sys.argv = _real_argv

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
REAL_OUTPUT_DIR = os.path.join(
    ENET_DIR, "finn_deployment_outputs",
    "12_dense_relu_warmstart150ep_alpha05_trained_rtl_mvau_8way_full_v4_512x512_20260923_223139",
)
PARTITION_IDX = 5
PART_BUILD_DIR = os.path.join(ENET_DIR, "finn_build_tmp", "GenericPartition_5_refix")
os.makedirs(PART_BUILD_DIR, exist_ok=True)
os.environ["FINN_BUILD_DIR"] = PART_BUILD_DIR

partition_fn = os.path.join(REAL_OUTPUT_DIR, "intermediate_models", "supported_op_partitions", "partition_5.onnx")
prefifo_ckpt = os.path.join(REAL_OUTPUT_DIR, "intermediate_models", "partition_5_prefifo_autosize_FIXED.onnx")

cfg = dataclasses.replace(buildmod.base.cfg_stitched_ip_partitioned_8way, output_dir=REAL_OUTPUT_DIR)

kernel_model = ModelWrapper(partition_fn)
print(f"[partition {PARTITION_IDX}] loaded fixed pre-fifo model: {len(kernel_model.graph.node)} nodes", flush=True)

kernel_model = buildmod.step_minimize_bit_width_standalone_thresh_aware(kernel_model, cfg)
kernel_model = buildmod.step_fix_weight_dtype_bipolar_bug(kernel_model, cfg)
kernel_model = buildmod.step_force_dsp(kernel_model, cfg)

print(f"[partition {PARTITION_IDX}] running step_hw_codegen...", flush=True)
kernel_model = buildmod.step_hw_codegen(kernel_model, cfg)
print(f"[partition {PARTITION_IDX}] running step_hw_ipgen...", flush=True)
kernel_model = buildmod.step_hw_ipgen(kernel_model, cfg)
print(f"[partition {PARTITION_IDX}] running step_set_fifo_depths (auto-sizing)...", flush=True)
kernel_model = buildmod.step_set_fifo_depths(kernel_model, cfg)

kernel_model.save(prefifo_ckpt)
print(f"[partition {PARTITION_IDX}] saved AUTOSIZED-FIFO (pre-split) checkpoint -> {prefifo_ckpt}", flush=True)

fifo_nodes = [n for n in kernel_model.graph.node if "StreamingFIFO" in n.op_type]
print(f"\n=== {len(fifo_nodes)} FIFO node(s) after autosizing ===")
total_depth = 0
rows = []
for n in fifo_nodes:
    inst = getCustomOp(n)
    depth = inst.get_nodeattr("depth")
    impl_style = inst.get_nodeattr("impl_style") if "impl_style" in inst.get_nodeattr_types() else "n/a"
    total_depth += depth
    rows.append((depth, n.name, n.op_type, impl_style))
for depth, name, op_type, impl_style in sorted(rows, reverse=True):
    print(f"  {name:40s} {op_type:20s} depth={depth:<8d} impl_style={impl_style}")
print(f"total FIFO depth (sum): {total_depth}")
print("\nNOT run: SplitLargeFIFOs / PrepareIP / HLSSynthIP / CreateStitchedIP / SynthOutOfContext")
print("Partition 6 and all other partitions were NOT touched by this script.")
