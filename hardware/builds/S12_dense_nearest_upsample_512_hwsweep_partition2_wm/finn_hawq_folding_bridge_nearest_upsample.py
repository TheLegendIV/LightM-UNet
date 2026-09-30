"""Bridge a MILP/finn_milp.py-produced layer_bits_folding_<tag>.json's
"per_layer" (pe/simd/thr_ram_style/...) values into a FINN
apply_folding_config-compatible JSON keyed by the ACTUAL generated node names
of partition 2 of the S12_dense_nearest_upsample_512_hwsweep_partition2_wm
sweep -- adapted from hardware/archive/finn_hawq_folding_bridge.py (the S19
original); see that file's docstring for the full rationale on WHY a bridge
is needed at all (FINN's post-specialize_layers node names have no direct
link back to the logical layer names the MILP solve was computed against).

Deltas vs. the S19 original:
  - CONV_ORDER_FILE points at this architecture's own conv-order dump
    (quantEnet_12_dense_relu_nearest_upsample_dummy_int8_conv_order.json --
    order is bit-width/checkpoint independent, dumped once from the dummy
    build, valid for every trained tag too).
  - Applies the SAME `_find_stage_boundaries_relaxed` monkeypatch the
    trained preamble script uses (StreamingMaxPool-based down1/down2,
    UpsampleNearestNeighbour-based up4/up5) BEFORE calling
    find_stage_boundaries -- required here since this is a fresh process
    and the preamble's own monkeypatch does not persist across processes.
  - FOLDING_BLOCK_FILE has no S19-specific default; always pass it explicitly
    (2nd CLI arg) -- MILP/finn_milp.py's per_layer entries observed for this
    sweep are all flat {"pe","simd",...} (no "node_type":"depthwise_vvau_slot"
    nesting), but that handling is kept dormant/harmless for forward
    compatibility.

Run inside the FINN container, AFTER finn_hawq_preamble_trained.py has
completed for the same tag:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_hawq_folding_bridge_nearest_upsample.py \\
        <hawq_preamble_output_dir> <layer_bits_folding_<tag>.json> [<out_config.json>]
Env ARMS_TARGET_FPS overrides the build config's target_fps (default 250).
"""
import dataclasses
import json
import math
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402

import finn_stage_partition  # noqa: E402


def _find_stage_boundaries_relaxed(model):
    """Verbatim copy of finn_hawq_preamble_trained.py's helper -- see that
    file's docstring for the full rationale. Must be applied here too since
    this is a separate process."""
    fmpad = model.get_nodes_by_op_type("FMPadding_Pixel")
    if len(fmpad) != 3:
        print(f"[find_stage_boundaries_relaxed] found {len(fmpad)} FMPadding_Pixel "
              "nodes (expected 3) -- ignoring, not used to derive boundaries.")
    maxpools = model.get_nodes_by_op_type("StreamingMaxPool")
    assert len(maxpools) in (2, 3), (
        "Expected 2 (down1, down2) or 3 (+ initial's own pool branch) "
        "StreamingMaxPool nodes, found %d." % len(maxpools)
    )
    node_index = lambda n: list(model.graph.node).index(n)  # noqa: E731
    maxpools = sorted(maxpools, key=node_index)
    down1_start = node_index(maxpools[-2])
    down2_start = node_index(maxpools[-1])

    upsample = [n for n in model.graph.node if n.op_type.startswith("UpsampleNearestNeighbour")]
    if len(upsample) == 2:
        upsample = sorted(upsample, key=node_index)
        up4_start = node_index(upsample[0])
        up5_start = node_index(upsample[1])
    else:
        print(f"[find_stage_boundaries_relaxed] found {len(upsample)} "
              "UpsampleNearestNeighbour_* nodes (expected 2) -- falling back "
              "to FMPadding_Pixel-pair detection for up4_start/up5_start.")
        fmpad_idx = sorted(node_index(n) for n in fmpad)
        groups = []
        for idx in fmpad_idx:
            if groups and idx - groups[-1][-1] <= 5:
                groups[-1].append(idx)
            else:
                groups.append([idx])
        pair_groups = [g for g in groups if len(g) >= 2]
        assert len(pair_groups) == 2, (
            "Expected exactly 2 FMPadding_Pixel pairs (up4.up, up5.up) as a "
            "fallback for missing Upsample nodes, found %d qualifying groups "
            "(all groups: %s)." % (len(pair_groups), groups)
        )
        up4_start, up5_start = pair_groups[0][0], pair_groups[1][0]

    boundaries = [down1_start, down2_start, up4_start, up5_start]
    assert boundaries == sorted(boundaries), (
        "Detected stage boundaries are not in ascending topological order (%s)" % boundaries
    )
    return boundaries


finn_stage_partition.find_stage_boundaries = _find_stage_boundaries_relaxed

# finn_enet_ip_build_partitioned_8way reads sys.argv[1]/[2] (MODEL_NAME/
# FIFO_STRATEGY) at import time -- mask our own CLI args during the import
# so they aren't misread as that module's positional args.
_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from finn_stage_partition import compute_8way_boundaries  # noqa: E402
from finn_partition_build_steps import step_create_dataflow_partition_multi  # noqa: E402
from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_specialize_layers,
    step_target_fps_parallelization,
)

PARTITION_IDX = 2
WEIGHT_OP_TYPES = ("MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl")
SWU_OP_TYPES = ("ConvolutionInputGenerator_hls", "ConvolutionInputGenerator_rtl")
FMPAD_OP_TYPES = ("FMPadding_hls", "FMPadding_rtl", "FMPadding_Pixel")
THRESH_OP_TYPES = ("Thresholding_hls", "Thresholding_rtl")
THRESH_DISTRIBUTED_BRAM_TRIGGER = 999999

CONV_ORDER_FILE = os.path.join(
    base.ENET_DIR, "quantEnet_12_dense_relu_nearest_upsample_dummy_int8_conv_order.json"
)

PARTITION_RANGE_ORDER = [
    "down1_start", "down2_start", "q2_start", "q3_start", "q4_start", "up4_start", "up5_start",
]


def partition_node_index_range(partition_idx, boundaries):
    edges = [0] + [boundaries[k] for k in PARTITION_RANGE_ORDER] + [None]
    lo = edges[partition_idx]
    hi = edges[partition_idx + 1]
    return lo, hi


def load_partition_logical_names(preamble_dir, partition_idx):
    pre_partition_ckpt = os.path.join(
        preamble_dir, "intermediate_models", "step_enet_convert_to_hw_rtl_mvau.onnx"
    )
    full_model = ModelWrapper(pre_partition_ckpt)

    # compute_8way_boundaries() is the single source of truth used by
    # assign_stage_partition_ids_8way itself (applies fork/non-HW boundary
    # shrinking on top of the raw find_stage_boundaries/
    # find_stage23_quarter_boundaries result) -- calling those two functions
    # directly here (as the S19 original bridge did) silently diverges from
    # the ACTUAL partition split whenever a shrink adjustment fires, which is
    # exactly what caused the 15-vs-14 node/name mismatch discovered when
    # validating this build.
    boundaries = compute_8way_boundaries(full_model)
    lo, hi = partition_node_index_range(partition_idx, boundaries)
    print(f"Partition {partition_idx} node-index range: [{lo}, {hi})  (boundaries={boundaries})")

    with open(CONV_ORDER_FILE) as f:
        all_names = json.load(f)  # full-network ordered logical names

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


def find_dense_swu_fmpad(kernel_model, mvau_node):
    """Walk producers upstream of a dense MVAU, skipping width converters, to
    find its (FMPadding or None, SWU) pair. Returns (None, None) when the MVAU
    has no SWU (1x1 conv). Dense convs need this because FINN leaves SWU SIMD
    at 1 unless it is set explicitly (finn_milp models it as simd_swu)."""
    def producer(node):
        p = kernel_model.find_producer(node.input[0])
        while p is not None and p.op_type.startswith("StreamingDataWidthConverter"):
            p = kernel_model.find_producer(p.input[0])
        return p
    swu = producer(mvau_node)
    if swu is None or swu.op_type not in SWU_OP_TYPES:
        return None, None
    fmpad = producer(swu)
    if fmpad is not None and fmpad.op_type not in FMPAD_OP_TYPES:
        fmpad = None
    return fmpad, swu


def find_following_thresholding(kernel_model, weight_node):
    """standalone Thresholding_hls/_rtl node directly consuming this MVAU/
    VVAU's output (noActivation=1 forced pre-partitioning, so activation is
    never fused into the weight node here). Ported from the golden v4/512x512
    reference -- see finn_gotchas.md."""
    consumer = kernel_model.find_consumer(weight_node.output[0])
    if consumer is not None and consumer.op_type in THRESH_OP_TYPES:
        return consumer
    return None


def main():
    if len(sys.argv) < 3:
        print("Usage: finn_hawq_folding_bridge_nearest_upsample.py <hawq_preamble_output_dir> <layer_bits_folding_json>")
        sys.exit(1)
    preamble_dir = sys.argv[1]
    folding_block_file = sys.argv[2]
    source_ckpt = os.path.join(preamble_dir, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
    print(f"Source checkpoint: {source_ckpt}")
    print(f"Folding block file: {folding_block_file}")

    flat_model = ModelWrapper(source_ckpt)
    cfg = base.cfg_stitched_ip_partitioned_8way
    if os.environ.get("ARMS_TARGET_FPS"):  # optional override (arms B/C experiment); default = cfg's own 250
        cfg = dataclasses.replace(cfg, target_fps=float(os.environ["ARMS_TARGET_FPS"]))
        print(f"target_fps overridden to {cfg.target_fps} via ARMS_TARGET_FPS")

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
    # model.transform(GiveUniqueNodeNames()) -- NO prefix -- right before
    # reading node.name; that unprefixed renaming OVERWRITES our prefixed
    # names above (same node order -> same generated names, no prefix). We
    # must generate our config keyed by THOSE names, not our own prefixed
    # ones, or every entry silently becomes a no-op.
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
    n_dense_swu = 0
    n_thresh = 0
    for node, logical_name in zip(weight_nodes, logical_names):
        entry, json_key = resolve_folding_entry(logical_name, per_layer)
        if entry is None:
            unmatched.append(logical_name)
            print(f"{node.name:30s} {node.op_type:12s} {logical_name:25s} {'<NO MATCH>':25s} {'':>4s} {'':>5s}")
            continue
        node_type = entry.get("node_type")  # None for these sweep's flat entries
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

        elif compute_entry.get("simd_swu") is not None and node.op_type.startswith("MVAU"):
            # Dense conv: MILP solves the SWU SIMD (simd_swu) separately from the MVAU SIMD.
            # Without this, FINN leaves SWU SIMD=1 and the SWU cycle count (~300k for stage2
            # convs) becomes the partition bottleneck, regardless of the MILP folding.
            fmpad_node, swu_node = find_dense_swu_fmpad(kernel_model, node)
            if swu_node is not None:
                simd_swu = compute_entry["simd_swu"]
                swu_cfg = {"SIMD": simd_swu}
                if swu_node.op_type == "ConvolutionInputGenerator_rtl":
                    # finn_cost_model.conv_cost_pe_simd: parallel_window iff MVAU SIMD > cin (then
                    # simd_swu = cin, swu_cycles = hin*win*cin/simd_swu + 2); else simd_swu = gcd(SIMD, cin).
                    # FINN leaves parallel_window=0 unless told, which prices the SWU ~K*K times slower.
                    ifm_ch = getCustomOp(swu_node).get_nodeattr("IFMChannels")
                    parallel_window = 1 if simd > ifm_ch else 0
                    swu_cfg["parallel_window"] = parallel_window
                    expected = ifm_ch if parallel_window else math.gcd(simd, ifm_ch)
                    if expected != simd_swu:
                        print(f"WARNING {swu_node.name}: MILP simd_swu={simd_swu} but cost-model rule gives {expected} "
                              f"(MVAU SIMD={simd}, IFMChannels={ifm_ch})")
                folding_config[swu_node.name] = swu_cfg
                n_dense_swu += 1
                extra = "".join(f" {k}={v}" for k, v in swu_cfg.items() if k != "SIMD")
                print(f"{swu_node.name:30s} {swu_node.op_type:12s} {'(SWU, simd_swu)':25s} {'':25s} {'':>4s} {simd_swu:5d}{extra}")
                if fmpad_node is not None:
                    folding_config[fmpad_node.name] = {"SIMD": simd_swu}
                    print(f"{fmpad_node.name:30s} {fmpad_node.op_type:12s} {'(FMPadding, simd_swu)':25s} {'':25s} {'':>4s} {simd_swu:5d}")

        # standalone Thresholding (noActivation=1 forced pre-partitioning)
        # gets its PE straight from this same entry's own "thr_pe" -- solved
        # jointly with pe/simd in the same ILP run. Ported from the golden
        # v4/512x512 reference (previously missing here -- every standalone
        # Thresholding_rtl node was silently left at FINN's auto-fold PE=1).
        thr_pe = compute_entry.get("thr_pe")
        if thr_pe is not None:
            thresh_node = find_following_thresholding(kernel_model, node)
            if thresh_node is not None:
                thr_config = {"PE": thr_pe}
                thr_ram_style = compute_entry.get("thr_ram_style")
                if thr_ram_style == "distributed":
                    thr_config["depth_trigger_bram"] = THRESH_DISTRIBUTED_BRAM_TRIGGER
                folding_config[thresh_node.name] = thr_config
                n_thresh += 1
                thr_extra = "".join(f" {k}={v}" for k, v in thr_config.items() if k != "PE")
                print(f"{thresh_node.name:30s} {thresh_node.op_type:12s} {'(thr_pe)':25s} {'':25s} {thr_pe:4d}{thr_extra}")

    if unmatched:
        print(f"\nWARNING: {len(unmatched)} logical names had no folding json entry: {unmatched}")
    if n_swu_fmpad:
        print(f"Bridged {n_swu_fmpad} FMPadding+SWU pair(s) for depthwise VVAU slots")
    if n_dense_swu:
        print(f"Bridged {n_dense_swu} SWU node(s) for dense MVAU convs via simd_swu")
    if n_thresh:
        print(f"Bridged {n_thresh} standalone Thresholding node(s) via thr_pe")

    # optional 3rd CLI arg: output path (lets several MILP foldings share one preamble dir)
    out_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(preamble_dir, "hawq_folding_config_partition2.json")
    with open(out_path, "w") as f:
        json.dump(folding_config, f, indent=2)
    print(f"\nSaved bridged folding config ({len(folding_config) - 1} node entries): {out_path}")


if __name__ == "__main__":
    main()
