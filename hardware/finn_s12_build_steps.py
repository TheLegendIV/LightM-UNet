"""Shared FINN build fixes + MILP folding bridge for S12-dense ENet builds
(decoder_type nearest_upsample / nearest_conv_upsample, 256x256 or 512x512).

Everything here was previously copy-pasted into every per-job build script;
see /memories/repo/finn_gotchas.md for the root cause behind each fix.
Container-only (needs FINN/QONNX); deployed flat next to the other shared infra.
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.datatype import DataType  # noqa: E402
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402
from qonnx.transformation.infer_datatypes import InferDataTypes  # noqa: E402
from qonnx.util.basic import calculate_matvec_accumulator_range, roundup_to_integer_multiple  # noqa: E402

import finn_stage_partition  # noqa: E402
from finn.builder.build_dataflow_steps import step_specialize_layers, step_target_fps_parallelization  # noqa: E402
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth  # noqa: E402
from finn.util.fpgadataflow import is_fpgadataflow_node  # noqa: E402

WEIGHT_OP_TYPES = ("MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl")
THRESH_OP_TYPES = ("Thresholding_hls", "Thresholding_rtl")
SWU_OP_TYPES = ("ConvolutionInputGenerator_hls", "ConvolutionInputGenerator_rtl")
FMPAD_OP_TYPES = ("FMPadding_hls", "FMPadding_rtl", "FMPadding_Pixel")
# Pinned Thresholding_rtl memory placement (thresholding.sv: stage depth >= trigger -> BRAM, else LUTRAM;
# 0 = Vivado "auto"). MUST match MILP/finn_cost_model.py THR_DEPTH_TRIGGER_BRAM / THR_DEPTH_TRIGGER_DISTRIBUTED,
# which is what the MILP priced: ram_style "block" -> 1024, "distributed" -> 999999 (everything LUTRAM).
THRESH_BRAM_TRIGGER = 1024
THRESH_DISTRIBUTED_BRAM_TRIGGER = 999999
THRESH_TRIGGER_BY_STYLE = {"block": THRESH_BRAM_TRIGGER, "distributed": THRESH_DISTRIBUTED_BRAM_TRIGGER}
PARTITION_RANGE_ORDER = [
    "down1_start", "down2_start", "q2_start", "q3_start", "q4_start", "up4_start", "up5_start",
]

# Ported from hardware/builds/12_dense_relu_nearest_conv_upsample_256_w8_16_v4's
# finn_ooc_..._8way_full_v4_256x256.py step_allocate_uram_fifos (opt-in here via
# finn_s12_build.py's --allocate-uram, default OFF -- "no URAM forcing" standing rule otherwise).
# ZCU7EV has 96 URAM288 blocks (4096-deep x 72b each) total; all 8 partitions are simultaneously
# resident on the SAME physical chip and share this ONE pool, so the whole-board budget MUST be
# discounted equally across the 8 partitions up front (OOC synth of one partition can't see the
# other 7's usage, so handing each partition the full board budget can oversubscribe the real chip
# by up to 8x). 88 (not the full 96) leaves a margin.
URAM_BUDGET_BLOCKS = 88
URAM_PARTITIONS = 8
URAM_DEPTH_PER_BLOCK = 4096
URAM_WIDTH_PER_BLOCK = 72


# ---------------------------------------------------------------- partitioning
def _find_stage_boundaries_relaxed(model):
    """StreamingMaxPool-based down1/down2, UpsampleNearestNeighbour-based up4/up5
    (FMPadding_Pixel-pair fallback)."""
    fmpad = model.get_nodes_by_op_type("FMPadding_Pixel")
    maxpools = model.get_nodes_by_op_type("StreamingMaxPool")
    assert len(maxpools) in (2, 3), "Expected 2 or 3 StreamingMaxPool nodes, found %d." % len(maxpools)
    node_index = lambda n: list(model.graph.node).index(n)  # noqa: E731
    maxpools = sorted(maxpools, key=node_index)
    down1_start = node_index(maxpools[-2])
    down2_start = node_index(maxpools[-1])

    upsample = [n for n in model.graph.node if n.op_type.startswith("UpsampleNearestNeighbour")]
    if len(upsample) == 2:
        upsample = sorted(upsample, key=node_index)
        up4_start, up5_start = node_index(upsample[0]), node_index(upsample[1])
    else:
        print(f"[find_stage_boundaries_relaxed] found {len(upsample)} UpsampleNearestNeighbour_* nodes "
              "(expected 2) -- falling back to FMPadding_Pixel-pair detection.")
        groups = []
        for idx in sorted(node_index(n) for n in fmpad):
            if groups and idx - groups[-1][-1] <= 5:
                groups[-1].append(idx)
            else:
                groups.append([idx])
        pair_groups = [g for g in groups if len(g) >= 2]
        assert len(pair_groups) == 2, "Expected 2 FMPadding_Pixel pairs, got groups %s" % groups
        up4_start, up5_start = pair_groups[0][0], pair_groups[1][0]

    boundaries = [down1_start, down2_start, up4_start, up5_start]
    assert boundaries == sorted(boundaries), "Stage boundaries not ascending (%s)" % boundaries
    return boundaries


def install_relaxed_stage_boundaries():
    """Must be called in every process that partitions or bridges (preamble AND build)."""
    finn_stage_partition.find_stage_boundaries = _find_stage_boundaries_relaxed


def check_dangling_nodes(model):
    graph_inputs = [i.name for i in model.graph.input]
    fwd_tensors, fwd_nodes = set(graph_inputs), set()
    queue = list(graph_inputs)
    while queue:
        for c in (model.find_consumers(queue.pop()) or []):
            if c.name not in fwd_nodes:
                fwd_nodes.add(c.name)
                for o in c.output:
                    if o not in fwd_tensors:
                        fwd_tensors.add(o)
                        queue.append(o)
    bwd_tensors, bwd_nodes = set(o.name for o in model.graph.output), set()
    queue = list(bwd_tensors)
    while queue:
        p = model.find_producer(queue.pop())
        if p is None or p.name in bwd_nodes:
            continue
        bwd_nodes.add(p.name)
        for i in p.input:
            if model.get_initializer(i) is None and i not in bwd_tensors:
                bwd_tensors.add(i)
                queue.append(i)
    dangling = [
        {"name": n.name, "op_type": n.op_type,
         "reachable_from_input": n.name in fwd_nodes, "reaches_output": n.name in bwd_nodes}
        for n in model.graph.node if not (n.name in fwd_nodes and n.name in bwd_nodes)
    ]
    return {"total_nodes": len(model.graph.node), "dangling_nodes": dangling, "n_dangling": len(dangling)}


# ------------------------------------------------------------- build-step fixes
def step_force_dsp(model, cfg=None):
    n = 0
    for node in model.graph.node:
        if "MVAU" in node.op_type or "VVAU" in node.op_type:
            getCustomOp(node).set_nodeattr("resType", "dsp")
            n += 1
    print(f"[step_force_dsp] forced resType=dsp on {n} MVAU/VVAU node(s)")
    return model


def step_fix_weight_dtype_bipolar_bug(model, cfg=None):
    """MinimizeWeightBitWidth picks BIPOLAR from weights.min() alone."""
    n_fixed = 0
    for node in model.graph.node:
        if node.op_type not in WEIGHT_OP_TYPES:
            continue
        inst = getCustomOp(node)
        if inst.get_nodeattr("weightDataType") != "BIPOLAR":
            continue
        w = model.get_initializer(node.input[1])
        if w is None or np.all((w == -1.0) | (w == 1.0)):
            continue
        w_min, w_max = float(w.min()), float(w.max())
        for cand in ["INT2", "INT3", "INT4", "INT5", "INT6", "INT7", "INT8"]:
            dt = DataType[cand]
            if dt.allowed(w_min) and dt.allowed(w_max):
                inst.set_nodeattr("weightDataType", cand)
                print(f"[step_fix_weight_dtype_bipolar_bug] {node.name}: BIPOLAR -> {cand} (w_min={w_min}, w_max={w_max})")
                n_fixed += 1
                break
        else:
            raise RuntimeError(f"{node.name}: could not find a valid INT dtype for w_min={w_min}, w_max={w_max}")
    if n_fixed:
        print(f"[step_fix_weight_dtype_bipolar_bug] fixed {n_fixed} mis-assigned BIPOLAR weightDataType node(s)")
    return model


def _widen_standalone_mvau_acc_for_downstream_threshold(inst, node, model):
    """FINN only widens the accumulator for FUSED thresholds; do the same using
    the downstream standalone Thresholding node's real threshold range."""
    if node.op_type not in WEIGHT_OP_TYPES:
        return False
    if len(node.input) > 2 or not inst.get_nodeattr("noActivation"):
        return False
    consumers = model.find_consumers(node.output[0]) or []
    if len(consumers) != 1 or consumers[0].op_type not in THRESH_OP_TYPES:
        return False
    thresh_node = consumers[0]
    thresholds = model.get_initializer(thresh_node.input[1])
    if thresholds is None:
        return False
    weights = model.get_initializer(node.input[1])
    if inst.get_nodeattr("binaryXnorMode"):
        weights = 2 * weights - 1
    acc_min, acc_max = calculate_matvec_accumulator_range(weights, inst.get_input_datatype())
    min_thr, max_thr = float(thresholds.min()), float(thresholds.max())
    if min_thr >= acc_min and max_thr <= acc_max:
        return False
    orig_min, orig_max = acc_min, acc_max
    acc_min, acc_max = min(acc_min, min_thr), max(acc_max, max_thr)
    if acc_min >= 0:
        adt = DataType[f"UINT{math.ceil(np.log2(acc_max + 1))}"]
    else:
        _acc_max = max(-acc_min, 1 + acc_max)
        adt = DataType[f"INT{math.ceil(np.log2(_acc_max) + 1)}"]
    if model.find_direct_successors(node) is None:
        bw = roundup_to_integer_multiple(adt.bitwidth(), 8)
        adt = DataType[adt.name.replace(str(adt.bitwidth()), str(bw))]
    inst.set_nodeattr("accDataType", adt.name)
    inst.set_nodeattr("outputDataType", adt.name)
    print(f"[standalone-threshold acc widen] {node.name}: acc range [{orig_min}, {orig_max}] too narrow for "
          f"{thresh_node.name}'s thresholds [{min_thr}, {max_thr}] -- set accDataType=outputDataType={adt.name}")
    return True


def step_minimize_bit_width_standalone_thresh_aware(model, cfg):
    """Drop-in replacement for FINN's step_minimize_bit_width."""
    if not cfg.minimize_bit_width:
        return model
    model = model.transform(MinimizeWeightBitWidth())
    n_widened = 0
    for node_id in range(len(model.graph.node)):
        node = model.graph.node[node_id]
        if not is_fpgadataflow_node(node):
            continue
        inst = getCustomOp(node)
        if not hasattr(inst, "minimize_accumulator_width"):
            continue
        if _widen_standalone_mvau_acc_for_downstream_threshold(inst, node, model):
            n_widened += 1
        else:
            inst.minimize_accumulator_width(model)
        model = model.transform(InferDataTypes())
    if n_widened:
        print(f"[step_minimize_bit_width_standalone_thresh_aware] widened {n_widened} MVAU/VVAU node(s)")
    return model


# --------------------------------------------------------------- folding bridge
def _largest_divisor_leq(n, cap):
    cap = max(1, min(cap, n))
    for d in range(cap, 0, -1):
        if n % d == 0:
            return d
    return 1


def _get_pe_simd_bounds(inst):
    try:
        return inst.get_nodeattr("MH"), inst.get_nodeattr("MW")
    except AttributeError:
        k_h, k_w = inst.get_nodeattr("Kernel")
        return inst.get_nodeattr("Channels"), k_h * k_w


REALIGN_WINDOW = 12  # how far ahead to search conv_order.json for a weight-count match on mismatch


def _shape_count(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def load_partition_logical_names(preamble_dir, conv_order_file, n_partitions=8):
    """{partition_idx: (conv_logical_names, pool_logical_names)} via positional
    match of conv_order.json against the pre-partition graph, shape-validated against each
    node's real weight element count. conv_order.json is built from plain PyTorch forward
    hooks (Python statement order), but the real exported graph's node order follows
    topological/readiness order instead -- these DIVERGE whenever a block forks into two
    branches of different depth before the next weight-bearing op (confirmed for
    FINNUpsamplingBottleneck's skip_resize_conv (3 hops: main_proj->main_act->main_up) vs.
    reduce (1 hop, same fork point) -- real graph order ends up [..., reduce, skip_resize_conv,
    ...], not conv_order.json's [..., skip_resize_conv, reduce, ...]). When the next unconsumed
    conv_order.json entry's weight_shape element count doesn't match the real node's weight
    initializer, search ahead (REALIGN_WINDOW) for the entry that does and swap it forward,
    rather than silently mis-assigning folding params to the wrong real conv. A forked MatMul
    duplicated by step_dedup_forked_matmul_before_threshold (same weight tensor) inherits the
    first copy's logical name."""
    full_model = ModelWrapper(f"{preamble_dir}/intermediate_models/step_enet_convert_to_hw_rtl_mvau.onnx")
    boundaries = finn_stage_partition.compute_8way_boundaries(full_model)
    print(f"[bridge] 8-way boundaries: {boundaries}")
    edges = [0] + [boundaries[k] for k in PARTITION_RANGE_ORDER] + [None]

    with open(conv_order_file) as f:
        all_names = json.load(f)

    weight_like_idx = [
        idx for idx, node in enumerate(full_model.graph.node)
        if node.op_type in ("MatrixVectorActivation", "MVAU", "VVAU") or "MaxPool" in node.op_type
    ]

    def _weight_tensor(node):
        for inp in node.input:
            if full_model.get_initializer(inp) is not None:
                return inp
        return None

    tensor_to_entry, node_idx_to_entry, pos = {}, {}, 0
    for node_idx in weight_like_idx:
        node = full_model.graph.node[node_idx]
        wt = _weight_tensor(node)
        if wt is not None and wt in tensor_to_entry:
            node_idx_to_entry[node_idx] = tensor_to_entry[wt]
            continue
        if pos >= len(all_names):
            raise RuntimeError(f"ran out of conv_order.json entries at node_idx={node_idx} -- do not proceed.")
        actual_count = None
        if wt is not None:
            arr = full_model.get_initializer(wt)
            if arr is not None:
                actual_count = int(np.prod(arr.shape))
        entry = all_names[pos]
        if (actual_count is not None and entry["weight_shape"] is not None
                and _shape_count(entry["weight_shape"]) != actual_count):
            match_j = next(
                (j for j in range(pos + 1, min(pos + 1 + REALIGN_WINDOW, len(all_names)))
                 if all_names[j]["weight_shape"] is not None and _shape_count(all_names[j]["weight_shape"]) == actual_count),
                None,
            )
            if match_j is None:
                raise RuntimeError(
                    f"node_idx={node_idx} ({node.name}): conv_order.json entry at pos={pos} "
                    f"({entry['logical_name']!r}, weight_shape={entry['weight_shape']}) doesn't match this "
                    f"node's real weight element count ({actual_count}), and no match found within the next "
                    f"{REALIGN_WINDOW} entries -- do not proceed."
                )
            print(f"[bridge] REALIGN: conv_order.json pos={pos} ({entry['logical_name']!r}) doesn't match "
                  f"node_idx={node_idx} ({node.name})'s real weight (count={actual_count}) -- swapping in "
                  f"pos={match_j} ({all_names[match_j]['logical_name']!r}) instead.")
            all_names[pos], all_names[match_j] = all_names[match_j], all_names[pos]
            entry = all_names[pos]
        pos += 1
        if wt is not None:
            tensor_to_entry[wt] = entry
        node_idx_to_entry[node_idx] = entry
    if pos != len(all_names):
        raise RuntimeError(f"consumed only {pos}/{len(all_names)} conv_order.json entries -- do not proceed.")

    result = {i: ([], []) for i in range(n_partitions)}
    for node_idx in weight_like_idx:
        pid = next(i for i in range(n_partitions)
                   if edges[i] <= node_idx and (edges[i + 1] is None or node_idx < edges[i + 1]))
        entry = node_idx_to_entry[node_idx]
        result[pid][1 if "MaxPool" in entry["module_type"] else 0].append(entry["logical_name"])
    return result


def _resolve_folding_entry(logical_name, per_layer):
    if logical_name in per_layer:
        return per_layer[logical_name], logical_name
    if logical_name.endswith(".conv.0") and logical_name[:-2] in per_layer:
        return per_layer[logical_name[:-2]], logical_name[:-2]
    return None, None


def _find_dense_swu_fmpad(kernel_model, mvau_node):
    """(FMPadding|None, SWU) feeding a dense MVAU, skipping DWCs; (None, None) for 1x1."""
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


def _find_preceding_swu_fmpad_vvau(kernel_model, vvau_node):
    swu = kernel_model.find_producer(vvau_node.input[0])
    if swu is None or swu.op_type not in SWU_OP_TYPES:
        raise RuntimeError(f"{vvau_node.name}: expected an SWU feeding it, got {getattr(swu, 'op_type', None)}")
    fmpad = kernel_model.find_producer(swu.input[0])
    if fmpad is not None and fmpad.op_type not in FMPAD_OP_TYPES:
        fmpad = None
    return fmpad, swu


def _find_following_thresholding(kernel_model, weight_node):
    consumer = kernel_model.find_consumer(weight_node.output[0])
    if consumer is not None and consumer.op_type in THRESH_OP_TYPES:
        return consumer
    return None


def _uram_blocks_for_fifo(inst, depth):
    """URAM288 tiling: each block is 4096-deep x 72b -- ceil(width/72) * ceil(depth/4096)
    blocks needed to hold one FIFO of this width/depth."""
    width = inst.get_instream_width()
    return math.ceil(width / URAM_WIDTH_PER_BLOCK) * math.ceil(depth / URAM_DEPTH_PER_BLOCK)


def step_allocate_uram_fifos(model, partition_idx, budget=None):
    """Greedily allocates URAM to the deepest StreamingFIFO_rtl nodes (impl_style=="vivado"
    only -- impl_style=="rtl"/Q_srl FIFOs are shallow <=256-deep SRL chains with no BRAM/URAM
    primitive at all) up to a hard budget of `budget` URAM288 blocks (4096x72b each on
    UltraScale+). `budget` defaults to this partition's equal share of the whole-board
    URAM_BUDGET_BLOCKS across URAM_PARTITIONS (NOT the full board budget -- see this module's
    URAM_BUDGET_BLOCKS comment).

    MUST run before SplitLargeFIFOs (confirmed via direct FINN source read of
    set_fifo_depths.py's SplitLargeFIFOs.apply(): a FIFO's ram_style nodeattr is copied VERBATIM
    onto every one of its split children) -- allocating here, pre-split, automatically keeps a
    whole deep FIFO "chain" together in one contiguous URAM allocation once it gets split.
    Deepest-first greedy selection -- a later (shallower) candidate is still tried if an earlier,
    deeper one didn't fit the remaining budget."""
    if budget is None:
        budget = URAM_BUDGET_BLOCKS // URAM_PARTITIONS
    candidates = []
    for node in model.graph.node:
        if node.op_type != "StreamingFIFO_rtl":
            continue
        inst = getCustomOp(node)
        if inst.get_nodeattr("impl_style") != "vivado":
            continue  # shallow SRL-only FIFO, no BRAM/URAM primitive exists for it
        depth = inst.get_nodeattr("depth")
        blocks = _uram_blocks_for_fifo(inst, depth)
        candidates.append((depth, blocks, node, inst))
    candidates.sort(key=lambda c: c[0], reverse=True)  # deepest first

    used = 0
    allocated = []
    for depth, blocks, node, inst in candidates:
        if used + blocks > budget:
            continue
        inst.set_nodeattr("ram_style", "ultra")
        used += blocks
        allocated.append((node.name, depth, blocks))

    print(f"[partition {partition_idx}] step_allocate_uram_fifos: allocated {used}/{budget} URAM288 block(s) "
          f"(per-partition share of the board's {URAM_BUDGET_BLOCKS}-block total across {URAM_PARTITIONS} "
          f"partitions) to {len(allocated)}/{len(candidates)} vivado-impl FIFO(s):")
    for name, depth, blocks in allocated:
        print(f"  {name:30s} depth={depth:8d} blocks={blocks}")
    return model


def _find_join_thresholds(kernel_model, logical_names, prev_block=None):
    """{Thresholding node name: (block, kind)} for the residual-join thresholds the MILP prices as
    extra nodes `<block>.skip_quant|residual_add|out_act` (finn_cost_model.md "Residual-join
    thresholds"). Matched structurally around each AddStreams: residual_add = Thresholding
    consuming the add's output; out_act = the Thresholding consuming that; skip_quant = Thresholding
    feeding an add input whose producer is NOT an MVAU/VVAU (an MVAU-fed one is that conv's own thr,
    handled via _find_following_thresholding). Ported from
    hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/
    finn_hawq_folding_bridge_nearest_upsample.py's find_join_thresholds."""
    adds = [n for n in kernel_model.graph.node if n.op_type.startswith("AddStreams")]
    blocks = [ln[: -len(".expand.0")] for ln in logical_names if ln.endswith(".expand.0")]
    if len(adds) != len(blocks):
        print(f"WARNING: {len(adds)} AddStreams vs {len(blocks)} expand.0 logical names in partition -- "
              "join thresholds left on Vivado auto placement")
        return {}

    def is_thr(n):
        return n is not None and n.op_type in THRESH_OP_TYPES

    found = {}
    # The partition opens with the PREVIOUS block's residual_add + out_act thresholds (their
    # AddStreams is in the upstream partition): Thresholding nodes before the first Dup, fed by no
    # node of this graph.
    first_thr = next((n for n in kernel_model.graph.node if is_thr(n) and kernel_model.find_producer(n.input[0]) is None), None)
    if first_thr is not None and prev_block is not None:
        found[first_thr.name] = (prev_block, "residual_add")
        nxt = kernel_model.find_consumer(first_thr.output[0])
        if is_thr(nxt):
            found[nxt.name] = (prev_block, "out_act")
    for add, block in zip(adds, blocks):
        res = kernel_model.find_consumer(add.output[0])
        if is_thr(res):
            found[res.name] = (block, "residual_add")
            out = kernel_model.find_consumer(res.output[0])
            if is_thr(out):
                found[out.name] = (block, "out_act")
        for inp in add.input:
            prod = kernel_model.find_producer(inp)
            if not is_thr(prod):
                continue
            prod_in = kernel_model.find_producer(prod.input[0])
            if prod_in is None or prod_in.op_type not in WEIGHT_OP_TYPES:
                found[prod.name] = (block, "skip_quant")
    return found


def _previous_block_name(conv_order_file, first_logical_name):
    """Block (`<block>` of `<block>.expand.0`) immediately before `first_logical_name` in the
    full (cross-partition) conv order."""
    with open(conv_order_file) as f:
        names = [e["logical_name"] for e in json.load(f)]
    for ln in reversed(names[: names.index(first_logical_name)]):
        if ln.endswith(".expand.0"):
            return ln[: -len(".expand.0")]
    return None


def build_partition_folding_config(partition_model_fn, sdp_node_name, logical_names, per_layer, cfg, tag="",
                                    extra_nodes=None, conv_order_file=None):
    """MILP per_layer (pe/simd/simd_swu/thr_pe/thr_ram_style) -> FINN folding
    config keyed by the node names step_apply_folding_config will see.
    extra_nodes (the MILP folding json's own "extra_nodes" dict) + conv_order_file (the FULL,
    cross-partition conv_order.json) are optional: when both given, also bridges the residual-join
    Thresholding nodes' (skip_quant/residual_add/out_act) depth_trigger_bram from their own
    ram_style -- without them, join thresholds are left on Vivado auto placement."""
    log = f"[bridge {tag}]"
    kernel_model = ModelWrapper(partition_model_fn)
    kernel_model = step_specialize_layers(kernel_model, cfg)
    kernel_model = kernel_model.transform(GiveUniqueNodeNames(sdp_node_name + "_"))
    kernel_model = kernel_model.transform(GiveReadableTensorNames())
    kernel_model = step_target_fps_parallelization(kernel_model, cfg)
    # step_apply_folding_config re-runs an UNPREFIXED GiveUniqueNodeNames before reading names.
    kernel_model = kernel_model.transform(GiveUniqueNodeNames())

    weight_nodes = [n for n in kernel_model.graph.node if n.op_type in WEIGHT_OP_TYPES]
    if len(weight_nodes) != len(logical_names):
        print(f"{log} MISMATCH -- FINN nodes: {[n.name + '/' + n.op_type for n in weight_nodes]}")
        print(f"{log} MISMATCH -- logical names: {logical_names}")
        raise RuntimeError(f"{log} weight node count != logical name count, aborting.")

    folding_config = {"Defaults": {}}
    unmatched = []
    for node, logical_name in zip(weight_nodes, logical_names):
        entry, json_key = _resolve_folding_entry(logical_name, per_layer)
        if entry is None:
            unmatched.append(logical_name)
            print(f"{log} {node.name:30s} {node.op_type:12s} {logical_name:25s} <NO MATCH>")
            continue
        compute_entry = entry["vvau"] if entry.get("node_type") == "depthwise_vvau_slot" else entry
        pe, simd = compute_entry["pe"], compute_entry["simd"]
        mh, mw = _get_pe_simd_bounds(getCustomOp(node))
        safe_pe, safe_simd = _largest_divisor_leq(mh, pe), _largest_divisor_leq(mw, simd)
        folding_config[node.name] = {"PE": safe_pe, "SIMD": safe_simd}
        note = "" if (safe_pe, safe_simd) == (pe, simd) else f" (clamped from PE={pe} SIMD={simd}, MH={mh} MW={mw})"
        print(f"{log} {node.name:30s} {node.op_type:12s} {logical_name:25s} [{json_key}] PE={safe_pe} SIMD={safe_simd}{note}")

        if entry.get("node_type") == "depthwise_vvau_slot":
            fmpad_node, swu_node = _find_preceding_swu_fmpad_vvau(kernel_model, node)
            folding_config[swu_node.name] = {"SIMD": safe_pe}
            if fmpad_node is not None:
                folding_config[fmpad_node.name] = {"SIMD": safe_pe}
        elif compute_entry.get("simd_swu") is not None and node.op_type.startswith("MVAU"):
            # FINN leaves dense-conv SWU SIMD=1 otherwise, making the SWU the bottleneck.
            fmpad_node, swu_node = _find_dense_swu_fmpad(kernel_model, node)
            if swu_node is not None:
                simd_swu = compute_entry["simd_swu"]
                if swu_node.op_type == "ConvolutionInputGenerator_rtl":
                    # Mirrors finn_cost_model.conv_cost_pe_simd: parallel_window iff MVAU SIMD > cin.
                    ifm_ch = getCustomOp(swu_node).get_nodeattr("IFMChannels")
                    parallel_window = 1 if simd > ifm_ch else 0
                    expected = ifm_ch if parallel_window else math.gcd(simd, ifm_ch)
                    if expected != simd_swu:
                        # select_impl_style() hard-asserts SIMD==IFMChannels when parallel_window=1
                        # on a dense conv -- clamp to the cost-model-safe value or FINN crashes.
                        print(f"{log} WARNING {swu_node.name}: MILP simd_swu={simd_swu} but cost-model rule gives "
                              f"{expected} -- clamping to {expected}")
                        simd_swu = expected
                    swu_cfg = {"SIMD": simd_swu, "parallel_window": parallel_window}
                else:
                    swu_cfg = {"SIMD": simd_swu}
                folding_config[swu_node.name] = swu_cfg
                print(f"{log} {swu_node.name:30s} {swu_node.op_type:12s} (SWU) {swu_cfg}")
                if fmpad_node is not None:
                    folding_config[fmpad_node.name] = {"SIMD": simd_swu}

        thr_pe = compute_entry.get("thr_pe")
        if thr_pe is not None:
            thresh_node = _find_following_thresholding(kernel_model, node)
            if thresh_node is not None:
                thr_config = {"PE": thr_pe}
                thr_ram_style = compute_entry.get("thr_ram_style")
                if thr_ram_style in THRESH_TRIGGER_BY_STYLE:
                    thr_config["depth_trigger_bram"] = THRESH_TRIGGER_BY_STYLE[thr_ram_style]
                folding_config[thresh_node.name] = thr_config
                print(f"{log} {thresh_node.name:30s} {thresh_node.op_type:12s} (thr_pe) {thr_config}")

    if unmatched:
        print(f"{log} WARNING: {len(unmatched)} logical names had no folding entry: {unmatched}")

    # residual-join thresholds: memory placement only (their MILP PE is not applied here -- they
    # stay at FINN's own PE, same as the original partition2_wm bridge this was ported from).
    n_join = 0
    if extra_nodes is not None and conv_order_file is not None:
        prev_block = _previous_block_name(conv_order_file, logical_names[0]) if logical_names else None
        joins = _find_join_thresholds(kernel_model, logical_names, prev_block)
        for thr_name, (block, kind) in joins.items():
            style = (extra_nodes.get(f"{block}.{kind}") or {}).get("ram_style")
            if style not in THRESH_TRIGGER_BY_STYLE:
                print(f"{log} WARNING {thr_name}: no ram_style for {block}.{kind} in folding json (got {style!r}) -- left on auto")
                continue
            folding_config.setdefault(thr_name, {})["depth_trigger_bram"] = THRESH_TRIGGER_BY_STYLE[style]
            n_join += 1
            print(f"{log} {thr_name:30s} Thresholding  (join {kind}) {block} ram_style={style} "
                  f"depth_trigger_bram={THRESH_TRIGGER_BY_STYLE[style]}")
        if n_join:
            print(f"{log} pinned depth_trigger_bram on {n_join} residual-join Thresholding node(s)")
    elif extra_nodes is not None or conv_order_file is not None:
        print(f"{log} WARNING: need BOTH extra_nodes and conv_order_file to bridge join thresholds -- got "
              f"extra_nodes={extra_nodes is not None} conv_order_file={conv_order_file is not None}, skipping")

    return folding_config
