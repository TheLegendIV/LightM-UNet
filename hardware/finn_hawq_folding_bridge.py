"""Bridge `compression/hawq/folding_block_s19.json`'s ILP-derived per-logical-
layer PE/SIMD values into a FINN `apply_folding_config`-compatible JSON keyed
by the ACTUAL generated node names of partition 2 (down2 + the first ~5.4
blocks of stage2, see below) of the HAWQ per-block S19 export, after
`step_specialize_layers`.

Why a bridge is needed: `folding_block_s19.json` was computed by an ILP cost
model against the LOGICAL layer names of the real ENet topology
(`enet/nnunetv2/nets/QuantENetS19Block.py`'s `BLOCK_NAMES`-style naming, e.g.
"stage2.1.conv.0"), but FINN's own node names after specialize_layers are
generated sequentially per op_type (e.g. "MVAU_hls_17") with no direct link
back to the logical name. This script re-establishes that link using the
ORDERED correspondence between:
  (a) `hardware/outputs/finn_exports/quantEnet_s19_hawq_block_int8_conv_order.json`
      -- ordered (named_modules() order == forward() order) logical names
      dumped straight from the PyTorch model (see finn_hawq_dump_conv_order.py)
  (b) the ordered list of MVAU_hls/MVAU_rtl/VVAU_hls/VVAU_rtl node names in
      partition 2's kernel model graph after step_specialize_layers (FINN
      does not reorder nodes during specialization, so this order matches
      the original Conv/ConvTranspose op order 1:1).

IMPORTANT (confirmed via finn_hawq_diag_partition_ranges.py): partition 2 of
the 8-way stage-based split is NOT "all of stage2/3" -- `assign_stage_partition_ids_8way`
cuts stage2/3's raw node-index range into 4 roughly-EQUAL-SIZED quarters (by
node count, not by logical block boundary), and the first quarter (partition_id=2)
ADDITIONALLY absorbs down2's downsample. For this HAWQ export specifically
(boundaries down1=7 down2=81 q2=186 q3=291 q4=396 up4=500 up5=551, confirmed
via the preamble's own build log), partition 2 == down2 + stage2.0..stage2.5
(partial -- only stage2.5.reduce.0, NOT its conv/expand). This script
re-derives that range PROGRAMMATICALLY (not hardcoded) using the exact same
`find_stage_boundaries`/`find_stage23_quarter_boundaries` helpers
`finn_stage_partition.py` itself uses, applied to the preamble's own saved
`step_enet_convert_to_hw.onnx` (pre-partition, full-graph) checkpoint -- so it
stays correct even if re-run against a differently-shaped export.

Normalization rule needed between (a) and folding_block_s19.json's keys:
non-dilated RegularBottleneck blocks (stage2.0/.5/.6/.11, stage3.* analog)
wrap their single conv in `nn.Sequential(conv, bn, act)`, so named_modules()
calls it "<name>.conv.0" -- but folding_block_s19.json's ILP model (which
didn't wrap it in a Sequential) calls it bare "<name>.conv". Dilated slots'
h/v separable convs ("<name>.conv.0"/"<name>.conv.3") already match exactly
in both. Try the exact key first, then the ".conv.0" -> ".conv" fallback.
Also down1/down2's `shortcut_proj.0` and `initial.pool` have NO
folding_block_s19.json entry at all (ILP cost model didn't cost them
separately / this FINN-safe topology has no real initial-block maxpool) --
these are left unmatched (skipped, printed as a warning) by design.

UPDATED for compression/hawq/folding_ilp.py's solve_folding_nodewise output
(--node-level): a per_layer entry may now carry a "node_type" field --
"mvau"/"maxpool" keep the flat {"pe", "simd"} shape resolve_folding_entry
already returns, but "depthwise_vvau_slot" nests the real compute node's
(PE, SIMD) under entry["vvau"] and additionally requires writing the SAME
PE onto the two FINN nodes immediately preceding the VVAU in node order
(FMPadding_*, then ConvolutionInputGenerator_*/SWU) -- see
find_preceding_swu_fmpad below and finn_ooc_..._8way_full.py's own copy of
this logic, which this bridge now mirrors. A legacy solve_folding (no
"node_type" key) folding_block_*.json still works unchanged (node_type is
None -> treated as flat/"mvau").

Run inside the FINN container, AFTER finn_hawq_preamble.py has completed:
    docker exec -e HOME=/tmp/home_dir <container> python3 \
        /home/thelegendiv/finn/notebooks/enet/finn_hawq_folding_bridge.py \
        <hawq_preamble_output_dir>
"""
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from finn_stage_partition import (  # noqa: E402
    find_stage_boundaries,
    find_stage23_quarter_boundaries,
)
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402

# finn_enet_ip_build_partitioned_8way reads sys.argv[1]/[2] (MODEL_NAME/
# FIFO_STRATEGY) at import time -- mask our own CLI args during the import
# so they aren't misread as that module's positional args.
_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from finn_partition_build_steps import step_create_dataflow_partition_multi  # noqa: E402
from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_specialize_layers,
    step_target_fps_parallelization,
)

PARTITION_IDX = 2
WEIGHT_OP_TYPES = ("MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl")
# Real FINN node types for the sliding-window unit and its preceding padding
# block, feeding a VVAU -- see compression/hawq/folding_ilp.py's
# solve_folding_nodewise, whose "depthwise_vvau_slot" entries this bridge
# now also writes onto these two node types.
SWU_OP_TYPES = ("ConvolutionInputGenerator_hls", "ConvolutionInputGenerator_rtl")
FMPAD_OP_TYPES = ("FMPadding_hls", "FMPadding_rtl", "FMPadding_Pixel")

CONV_ORDER_FILE = os.path.join(
    base.ENET_DIR, "quantEnet_s19_hawq_block_int8_conv_order.json"
)
FOLDING_BLOCK_FILE = os.path.join(
    base.ENET_DIR, "folding_block_s19.json"
)


# node-index range -> partition_id, mirrors assign_stage_partition_ids_8way's
# own branching exactly (see finn_stage_partition.py).
PARTITION_RANGE_ORDER = [
    "down1_start", "down2_start", "q2_start", "q3_start", "q4_start", "up4_start", "up5_start",
]


def partition_node_index_range(partition_idx, boundaries):
    """boundaries = dict with keys down1_start/down2_start/q2_start/q3_start/
    q4_start/up4_start/up5_start (see assign_stage_partition_ids_8way).
    Returns (lo, hi) node-index range (hi exclusive) for the given
    partition_id (0..7)."""
    edges = [0] + [boundaries[k] for k in PARTITION_RANGE_ORDER] + [None]
    lo = edges[partition_idx]
    hi = edges[partition_idx + 1]
    return lo, hi


def load_partition_logical_names(preamble_dir, partition_idx):
    """Re-derive the exact ordered list of logical conv/pool names belonging
    to `partition_idx`, by intersecting node-index ranges (recomputed fresh
    from step_enet_convert_to_hw.onnx, same as assign_stage_partition_ids_8way
    itself does) against the full ordered logical-name list. Returns
    (conv_names, skipped_pool_names) -- pool names (StreamingMaxPool, e.g.
    down2.shortcut_pool) have no PE/SIMD-style folding entry shape and are
    reported separately, not zipped against MVAU/VVAU nodes."""
    pre_partition_ckpt = os.path.join(
        preamble_dir, "intermediate_models", "step_enet_convert_to_hw.onnx"
    )
    full_model = ModelWrapper(pre_partition_ckpt)

    down1_start, down2_start, up4_start, up5_start = find_stage_boundaries(full_model)
    q2_start, q3_start, q4_start = find_stage23_quarter_boundaries(down2_start, up4_start)
    boundaries = {
        "down1_start": down1_start, "down2_start": down2_start,
        "q2_start": q2_start, "q3_start": q3_start, "q4_start": q4_start,
        "up4_start": up4_start, "up5_start": up5_start,
    }
    lo, hi = partition_node_index_range(partition_idx, boundaries)
    print(f"Partition {partition_idx} node-index range: [{lo}, {hi})  (boundaries={boundaries})")

    with open(CONV_ORDER_FILE) as f:
        all_names = json.load(f)  # full-network ordered logical names (129 entries for S19 HAWQ)

    weight_like_idx = [
        idx for idx, node in enumerate(full_model.graph.node)
        if node.op_type in ("MatrixVectorActivation", "MVAU", "VVAU") or "MaxPool" in node.op_type
    ]
    if len(weight_like_idx) != len(all_names):
        raise RuntimeError(
            f"weight-like node count in pre-partition graph ({len(weight_like_idx)}) != "
            f"logical name list length ({len(all_names)}) -- positional correspondence broken, "
            "do not proceed."
        )

    conv_names, pool_names = [], []
    for pos, node_idx in enumerate(weight_like_idx):
        if not (lo <= node_idx < hi):
            continue
        entry = all_names[pos]
        if "MaxPool" in entry["module_type"]:
            pool_names.append(entry["logical_name"])
        else:
            conv_names.append(entry["logical_name"])
    return conv_names, pool_names


def resolve_folding_entry(logical_name, per_layer):
    if logical_name in per_layer:
        return per_layer[logical_name], logical_name
    if logical_name.endswith(".conv.0"):
        stripped = logical_name[: -len(".0")]
        if stripped in per_layer:
            return per_layer[stripped], stripped
    return None, None


def find_preceding_swu_fmpad(all_nodes, name_to_idx, vvau_node):
    """Real FINN's step_specialize_layers always emits a depthwise conv's
    three nodes back-to-back, in the fixed order [FMPadding_i,
    ConvolutionInputGenerator_i, VVAU_j] -- see finn_ooc_..._8way_full.py's
    identical helper for the full rationale."""
    idx = name_to_idx[vvau_node.name]
    if idx < 2:
        raise RuntimeError(f"{vvau_node.name}: expected 2 preceding nodes (FMPadding, SWU), only {idx} nodes before it")
    swu_node, fmpad_node = all_nodes[idx - 1], all_nodes[idx - 2]
    if swu_node.op_type not in SWU_OP_TYPES:
        raise RuntimeError(f"{vvau_node.name}: expected an SWU node immediately before it, got "
                            f"{swu_node.op_type} ({swu_node.name})")
    if fmpad_node.op_type not in FMPAD_OP_TYPES:
        raise RuntimeError(f"{vvau_node.name}: expected an FMPadding node 2 positions before it, got "
                            f"{fmpad_node.op_type} ({fmpad_node.name})")
    return fmpad_node, swu_node


def main():
    if len(sys.argv) < 2:
        print("Usage: finn_hawq_folding_bridge.py <hawq_preamble_output_dir> [folding_block_json_path]")
        sys.exit(1)
    preamble_dir = sys.argv[1]
    folding_block_file = sys.argv[2] if len(sys.argv) > 2 else FOLDING_BLOCK_FILE
    source_ckpt = os.path.join(preamble_dir, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
    print(f"Source checkpoint: {source_ckpt}")
    print(f"Folding block file: {folding_block_file}")

    flat_model = ModelWrapper(source_ckpt)
    cfg = base.cfg_stitched_ip_partitioned_8way

    print("Running step_create_dataflow_partition_multi (re-split, deterministic)...")
    parent_model = step_create_dataflow_partition_multi(flat_model, cfg)
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    print(f"Got {len(sdp_nodes)} partitions: {[n.name for n in sdp_nodes]}")
    sdp_node = sdp_nodes[PARTITION_IDX]
    sdp_inst = getCustomOp(sdp_node)
    partition_model_fn = sdp_inst.get_nodeattr("model")
    print(f"Partition {PARTITION_IDX} -> {sdp_node.name} -> {partition_model_fn}")

    prefix = sdp_node.name + "_"
    kernel_model = ModelWrapper(partition_model_fn)
    print(f"Loaded raw partition {PARTITION_IDX} model: {len(kernel_model.graph.node)} nodes")

    print("Running: step_specialize_layers")
    kernel_model = step_specialize_layers(kernel_model, cfg)
    kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
    kernel_model = kernel_model.transform(GiveReadableTensorNames())
    print("Running: step_target_fps_parallelization")
    kernel_model = step_target_fps_parallelization(kernel_model, cfg)
    # step_apply_folding_config (in the REAL build) calls
    # model.transform(GiveUniqueNodeNames()) -- NO prefix -- internally,
    # RIGHT BEFORE reading node.name to look up each node's config entry.
    # That unprefixed renaming pass OVERWRITES whatever prefix we set above
    # (same node order -> same generated names, just without the
    # "GenericPartition_2_" prefix). We must generate our config keyed by
    # THOSE exact (unprefixed) names, not our own prefixed ones, or every
    # entry silently becomes an "unused HW configuration" no-op.
    kernel_model = kernel_model.transform(GiveUniqueNodeNames())

    weight_nodes = [n for n in kernel_model.graph.node if n.op_type in WEIGHT_OP_TYPES]
    print(f"Found {len(weight_nodes)} weight-bearing nodes (op_type in {WEIGHT_OP_TYPES}) in partition {PARTITION_IDX}")

    logical_names, pool_names = load_partition_logical_names(preamble_dir, PARTITION_IDX)
    print(f"Found {len(logical_names)} logical conv names for partition {PARTITION_IDX}: {logical_names}")
    if pool_names:
        print(f"(skipped {len(pool_names)} pool-type logical names, no PE/SIMD entry shape: {pool_names})")

    if len(weight_nodes) != len(logical_names):
        print("MISMATCH: counts differ -- do NOT proceed blindly. Printing both lists for inspection:")
        print("-- FINN weight nodes (order) --")
        for n in weight_nodes:
            print(f"  {n.name}  ({n.op_type})")
        print("-- Logical names (order) --")
        for ln in logical_names:
            print(f"  {ln}")
        sys.exit(2)

    with open(folding_block_file) as f:
        folding_block = json.load(f)
    per_layer = folding_block["per_layer"]

    all_nodes = list(kernel_model.graph.node)
    name_to_idx = {n.name: i for i, n in enumerate(all_nodes)}

    folding_config = {"Defaults": {}}
    print(f"{'FINN node':30s} {'op_type':12s} {'logical name':25s} {'json key':25s} {'PE':>4s} {'SIMD':>5s}")
    unmatched = []
    n_swu_fmpad = 0
    for node, logical_name in zip(weight_nodes, logical_names):
        entry, json_key = resolve_folding_entry(logical_name, per_layer)
        if entry is None:
            unmatched.append(logical_name)
            print(f"{node.name:30s} {node.op_type:12s} {logical_name:25s} {'<NO MATCH>':25s} {'':>4s} {'':>5s}")
            continue
        node_type = entry.get("node_type")  # None for a legacy solve_folding-shaped entry
        compute_entry = entry["vvau"] if node_type == "depthwise_vvau_slot" else entry
        pe, simd = compute_entry["pe"], compute_entry["simd"]
        folding_config[node.name] = {"PE": pe, "SIMD": simd}
        print(f"{node.name:30s} {node.op_type:12s} {logical_name:25s} {json_key:25s} {pe:4d} {simd:5d}")

        if node_type == "depthwise_vvau_slot":
            fmpad_node, swu_node = find_preceding_swu_fmpad(all_nodes, name_to_idx, node)
            folding_config[swu_node.name] = {"SIMD": pe}
            folding_config[fmpad_node.name] = {"SIMD": pe}
            n_swu_fmpad += 1
            print(f"{swu_node.name:30s} {swu_node.op_type:12s} {'(SWU, coupled)':25s} {'':25s} {'':>4s} {pe:5d}")
            print(f"{fmpad_node.name:30s} {fmpad_node.op_type:12s} {'(FMPadding, coupled)':25s} {'':25s} {'':>4s} {pe:5d}")

    if unmatched:
        print(f"\nWARNING: {len(unmatched)} logical names had no folding json entry: {unmatched}")
    if n_swu_fmpad:
        print(f"Bridged {n_swu_fmpad} FMPadding+SWU pair(s) for depthwise VVAU slots")

    out_path = os.path.join(preamble_dir, "hawq_folding_config_partition2.json")
    with open(out_path, "w") as f:
        json.dump(folding_config, f, indent=2)
    print(f"\nSaved bridged folding config ({len(folding_config) - 1} node entries): {out_path}")


if __name__ == "__main__":
    main()
