"""Preamble for finn_export_probe_8x_bottleneck_min_int4.py's tiny
8-bottleneck network -- produces the exact checkpoint/sidecar files
hardware/finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_8way_full.py
needs to run UNMODIFIED against this toy network (see
finn_test_8way_full_against_tiny_8x_bottleneck.py, the driver that then
calls that real script's own main()):

    <output_dir>/intermediate_models/step_enet_convert_to_hw.onnx
        (pre-partition checkpoint -- _8way_full.py's own
        load_all_partition_logical_names() reads this exact filename)
    <output_dir>/intermediate_models/assign_stage_partition_ids_8way.onnx
        (post-partition-id checkpoint -- _8way_full.py's own main() reads
        this exact filename as its flat_ckpt)
    <output_dir>/conv_order.json / folding_block.json
        (sidecars -- point _8way_full.py's CONV_ORDER_FILE/FOLDING_BLOCK_FILE
        at these; PE=SIMD=1 everywhere, weight_bits=4, matching this probe's
        W4A4/"min SIMD+PE" spec)

Uses FINN's OWN stock build steps (step_qonnx_to_finn/step_tidy_up/
step_streamline/step_convert_to_hw with cfg.standalone_thresholds=True) --
this toy network has none of the real S12 architecture's quirks (no Concat
initial block, no leaky activations, no down/upsampling), so none of
finn_enet_build_fixups.py's ENet-specific fixup passes are needed; the
stock pipeline handles it directly. cfg.standalone_thresholds=True is the
stock equivalent of finn_enet_convert_to_hw_rtl_mvau.py's manual
InferThresholdingLayer-before-MVAU-fusion trick (see that file's own
docstring) -- forces noActivation=1 on every MVAU, a prerequisite for RTL
specialization.

step_enet_convert_to_hw/assign_stage_partition_ids_8way below are thin
wrappers/replacements with those EXACT names (not the stock/real ones) so
FINN's own save_intermediate_models machinery (uses step_fn.__name__) writes
the checkpoints at the filenames _8way_full.py expects, with zero edits to
that file.

Partition boundaries are detected structurally via DuplicateStreams node
count (1 per bottleneck's own residual fork, first HW-visible op of each
block) -- the flat-topology equivalent of finn_stage_partition.py's own
StreamingMaxPool-count trick for down1/down2.

Usage (inside the FINN container, after docker cp-ing both this file and
finn_export_probe_8x_bottleneck_min_int4.py's OUTPUT onnx into
finn/notebooks/enet/):
    python3 finn_preamble_probe_8x_bottleneck_min_int4.py
"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import SortGraph
from qonnx.util.basic import get_by_name

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.build_dataflow_steps import (
    step_qonnx_to_finn,
    step_tidy_up,
    step_streamline,
    step_convert_to_hw,
)
from finn_enet_build_fixups import (
    step_absorb_leftover_scale_before_matmul,
    step_fuse_forked_dequant_into_duplicate_threshold,
)

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
MODEL_NAME = "quant_probe_8x_bottleneck_min_int4"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")
FPGA_PART = "xczu7ev-ffvc1156-2-e"

N_BLOCKS = 8
BIT_WIDTH = 4
STAGE_NAMES = ("reduce", "conv", "expand")  # forward-pass order within one QuantRegularBottleneck

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", f"probe_8x_bottleneck_min_int4_preamble_{timestamp}")


def _is_fpgadataflow_node(node):
    """Mirrors finn_stage_partition.py's own helper -- only fpgadataflow-
    backed HW nodes get a partition_id; leftover non-HW nodes are skipped."""
    backend = get_by_name(node.attribute, "backend")
    return backend is not None and backend.s.decode("UTF-8") == "fpgadataflow"


def step_fixup_residual_transpose_mul(model, cfg):
    """Reuses finn_enet_build_fixups.py's own proven fork/forked-dequant
    fixups (already solved for the real S12 architecture's DecomposedPReLU
    residual-adjacent fork) instead of a new bespoke transform -- this toy
    net's residual join (MultiThreshold -> Transpose -> Mul -> Add, on both
    operands) is the exact same class of problem: a dequant Mul stranded on
    one branch of a forked MultiThreshold, blocking HW AddStreams/
    DuplicateStreams conversion (confirmed via probe run: 0 DuplicateStreams
    found post step_convert_to_hw, 8 stray float Add + 8 orphaned
    MultiThreshold nodes)."""
    model = step_absorb_leftover_scale_before_matmul(model, cfg)
    model = step_fuse_forked_dequant_into_duplicate_threshold(model, cfg)
    return model


def step_enet_convert_to_hw(model, cfg):
    """Stock step_convert_to_hw, renamed so save_intermediate_models writes
    step_enet_convert_to_hw.onnx (the exact filename _8way_full.py's
    load_all_partition_logical_names() reads)."""
    return step_convert_to_hw(model, cfg)


def assign_stage_partition_ids_8way(model, cfg=None):
    """Flat-topology partitioner: N_BLOCKS DuplicateStreams nodes (one per
    bottleneck's own residual fork) mark the start of each bottleneck;
    partition_id = how many DuplicateStreams nodes precede this node's own
    index. Same name as finn_stage_partition.py's real (stage-boundary-
    based) function so the saved checkpoint matches _8way_full.py's
    expected assign_stage_partition_ids_8way.onnx filename -- NOT the same
    implementation, this network has no down/up-sampling stages."""
    model = model.transform(SortGraph())
    dup_idx = [i for i, n in enumerate(model.graph.node) if n.op_type == "DuplicateStreams"]
    assert len(dup_idx) == N_BLOCKS, (
        f"expected {N_BLOCKS} DuplicateStreams nodes (one per bottleneck's residual fork), "
        f"found {len(dup_idx)} -- topology assumption broken, do not proceed."
    )
    boundaries = dup_idx[1:]  # start index of bottleneck 1..N_BLOCKS-1 (bottleneck 0 starts at node 0)

    counts = [0] * N_BLOCKS
    n_skipped = 0
    for idx, node in enumerate(model.graph.node):
        if not _is_fpgadataflow_node(node):
            n_skipped += 1
            continue
        pid = sum(1 for b in boundaries if idx >= b)
        getCustomOp(node).set_nodeattr("partition_id", pid)
        counts[pid] += 1
    print(f"[assign_stage_partition_ids_8way] boundaries={boundaries}")
    print(f"[assign_stage_partition_ids_8way] node counts per partition: {counts} (skipped non-HW={n_skipped})")
    return model


def write_sidecars(output_dir):
    """conv_order.json + folding_block.json, derived from the just-saved
    step_enet_convert_to_hw.onnx checkpoint -- weight-like nodes (generic,
    unspecialized "MVAU" op_type at this pre-specialize stage) walked in
    topological order, 3 per bottleneck (reduce/conv/expand), named
    blocks.<i>.<stage> to match _8way_full.py's resolve_folding_entry
    (exact-key lookup, no special-casing needed since these names have no
    trailing ".0")."""
    ckpt = os.path.join(output_dir, "intermediate_models", "step_enet_convert_to_hw.onnx")
    model = ModelWrapper(ckpt)
    weight_nodes = [n for n in model.graph.node if n.op_type in ("MatrixVectorActivation", "MVAU", "VVAU")]
    assert len(weight_nodes) == N_BLOCKS * len(STAGE_NAMES), (
        f"expected {N_BLOCKS * len(STAGE_NAMES)} weight-like nodes ({N_BLOCKS} blocks x "
        f"{len(STAGE_NAMES)} stages), found {len(weight_nodes)} -- check the exported topology."
    )

    conv_order = []
    per_layer = {}
    pos = 0
    for i in range(N_BLOCKS):
        for stage in STAGE_NAMES:
            logical_name = f"blocks.{i}.{stage}"
            conv_order.append({"logical_name": logical_name, "module_type": "Conv2d"})
            per_layer[logical_name] = {"pe": 1, "simd": 1, "weight_bits": BIT_WIDTH}
            pos += 1
    assert pos == len(weight_nodes)

    conv_order_path = os.path.join(output_dir, "conv_order.json")
    folding_block_path = os.path.join(output_dir, "folding_block.json")
    with open(conv_order_path, "w") as f:
        json.dump(conv_order, f, indent=2)
    with open(folding_block_path, "w") as f:
        json.dump({"per_layer": per_layer}, f, indent=2)
    print(f"Wrote {conv_order_path} ({len(conv_order)} entries)")
    print(f"Wrote {folding_block_path} ({len(per_layer)} per_layer entries, PE=SIMD=1, weight_bits={BIT_WIDTH})")
    return conv_order_path, folding_block_path


steps = [
    step_qonnx_to_finn,
    step_tidy_up,
    step_streamline,
    step_fixup_residual_transpose_mul,
    step_enet_convert_to_hw,
    assign_stage_partition_ids_8way,
]

cfg = DataflowBuildConfig(
    output_dir          = OUTPUT_DIR,
    synth_clk_period_ns = 10.0,
    fpga_part           = FPGA_PART,
    standalone_thresholds = True,
    steps               = steps,
    generate_outputs    = [],
    save_intermediate_models = True,
)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Model : {MODEL_FILE}")
    print(f"Output: {OUTPUT_DIR}", flush=True)
    build.build_dataflow_cfg(MODEL_FILE, cfg)
    conv_order_path, folding_block_path = write_sidecars(OUTPUT_DIR)
    print("\nDone. Preamble dir:", OUTPUT_DIR)
    print("CONV_ORDER_FILE   =", conv_order_path)
    print("FOLDING_BLOCK_FILE=", folding_block_path)
