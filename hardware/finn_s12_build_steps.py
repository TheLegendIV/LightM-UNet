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
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC  # noqa: E402
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO  # noqa: E402
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth  # noqa: E402
from finn.transformation.fpgadataflow.set_fifo_depths import reset_implementation  # noqa: E402
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers  # noqa: E402
from finn.util.fpgadataflow import is_fpgadataflow_node  # noqa: E402

WEIGHT_OP_TYPES = ("MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl")
THRESH_OP_TYPES = ("Thresholding_hls", "Thresholding_rtl")
SWU_OP_TYPES = ("ConvolutionInputGenerator_hls", "ConvolutionInputGenerator_rtl")
FMPAD_OP_TYPES = ("FMPadding_hls", "FMPadding_rtl", "FMPadding_Pixel", "FMPadding_Pixel_hls", "FMPadding_Pixel_rtl")
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


def step_insert_argmax_output(model, cfg=None):
    """Append a per-pixel top-1 (argmax over the 5 class channels) to the real
    network output, converting `final`'s 5-channel logit map (the standalone
    ChannelwiseOp bias-add after `final`'s MVAU) into a 1-channel class-index
    map in-PL via FINN's LabelSelect HW op, instead of exporting raw logits
    for CPU-side argmax. Must run after convert_to_hw (LabelSelect only comes
    from lowering a TopK node) and before stage-partition assignment (so the
    new node's topological position places it in partition 7 alongside
    up5/regular5/final, same as assign_stage_partition_ids_8way's own
    index-based `else: pid = 7` fallback already does for any trailing node).

    At this pipeline stage the DECLARED graph output is still float32/NCHW
    (matching the original PyTorch export's I/O convention) -- convert_to_hw
    leaves a Transpose(NHWC->NCHW)+Mul(dequant scale) "glue" pair between the
    real last HW node (ChannelwiseOp_0, raw-integer NHWC) and graph.output
    (validated 2026-10-05: naively running InsertTopK against graph.output
    directly argmaxes over the Mul's NCHW tensor -- wrong axis, picks over W
    instead of channels). Walk back past that glue first; argmax must consume
    the real HW tensor directly, and the float rescale is meaningless once
    the output is a discrete class index, so the glue is dropped afterward.

    qonnx's InsertTopK also pops whatever tensor was graph.output[0]'s
    ValueInfoProto out of graph.output (to replace it with the new TopK
    indices output) without re-registering it anywhere else, leaving it
    shape/dtype-less for the very next transform (InferLabelSelectLayer) that
    needs to read them back off the now-intermediate tensor -- restored
    explicitly below before InferLabelSelectLayer runs."""
    from qonnx.transformation.insert_topk import InsertTopK
    from qonnx.transformation.infer_shapes import InferShapes
    from finn.transformation.fpgadataflow.convert_to_hw_layers import InferLabelSelectLayer

    real_out = model.graph.output[0].name
    glue_nodes = []
    producer = model.find_producer(real_out)
    while producer is not None and producer.domain == "":
        glue_nodes.append(producer)
        real_out = producer.input[0]
        producer = model.find_producer(real_out)

    # repoint graph.output at the real HW tensor, reusing its existing (correctly-shaped/
    # NHWC-laid-out) value_info entry so InsertTopK's axis=-1 targets the channel dim
    vi_idx = next(i for i, vi in enumerate(model.graph.value_info) if vi.name == real_out)
    del model.graph.output[0]
    model.graph.output.insert(0, model.graph.value_info.pop(vi_idx))

    out_shape = model.get_tensor_shape(real_out)
    out_dtype = model.get_tensor_datatype(real_out)
    model = model.transform(InsertTopK(k=1, axis=-1))
    model.set_tensor_shape(real_out, out_shape)
    model.set_tensor_datatype(real_out, out_dtype)
    model = model.transform(InferLabelSelectLayer())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    for n in glue_nodes:
        model.graph.node.remove(n)

    n_labelselect = sum(1 for node in model.graph.node if node.op_type == "LabelSelect")
    print(f"[step_insert_argmax_output] inserted {n_labelselect} LabelSelect node(s), "
          f"dropped {len(glue_nodes)} float-dequant glue node(s) ({[n.op_type for n in glue_nodes]}); "
          f"new output shape={model.get_tensor_shape(model.graph.output[0].name)}")
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


def match_conv_order_to_nodes(full_model, conv_order_file):
    """(weight_like_idx, {node_idx: conv_order entry}): the positional match of conv_order.json against the weight-bearing nodes (MVAU / VVAU / MaxPool) of a
    pre-partition graph, with the weight-count realignment described in load_partition_logical_names. Shared by the 8-way bridge and the per-block partitioning
    (finn_s12_blocks.py)."""
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

    return weight_like_idx, node_idx_to_entry


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

    weight_like_idx, node_idx_to_entry = match_conv_order_to_nodes(full_model, conv_order_file)

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


def _find_up_block_fmpad_chain(kernel_model, mvau_node):
    """up_bottleneck.py's 2x2-lowered transposed conv has TWO FMPadding nodes in series before its SWG --
    the zero-insertion 'FMPadPix' (op_type FMPadding_Pixel_hls, role 'fmpadpix') followed by the border
    'FMPad_u' (op_type FMPadding_rtl, role 'fmpad_u') -- unlike every other conv's single FMPadding, so
    _find_dense_swu_fmpad's one-hop walk only ever finds the nearer FMPad_u and mislabels it 'fmpadpix'.
    Returns (fmpadpix_node|None, fmpad_u_node|None, swu_node|None)."""
    def producer(node):
        p = kernel_model.find_producer(node.input[0])
        while p is not None and p.op_type.startswith("StreamingDataWidthConverter"):
            p = kernel_model.find_producer(p.input[0])
        return p
    swu = producer(mvau_node)
    if swu is None or swu.op_type not in SWU_OP_TYPES:
        return None, None, None
    fmpad_u = producer(swu)
    if fmpad_u is not None and fmpad_u.op_type not in FMPAD_OP_TYPES:
        fmpad_u = None
    fmpadpix = producer(fmpad_u) if fmpad_u is not None else None
    if fmpadpix is not None and fmpadpix.op_type not in FMPAD_OP_TYPES:
        fmpadpix = None
    return fmpadpix, fmpad_u, swu


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


def _claimed_mvau_names(kernel_model, logical_names, per_layer):
    """Weight-op node names build_partition_role_nodes resolves to a main-path leaf role (reduce.0/
    conv/conv.0/expand.0/main_proj.0/up.0/final) -- lets _find_join_thresholds tell a block's own
    main-path MVAU (whose Thresholding is already role-tagged via _find_following_thresholding) apart
    from an MVAU-fed Thresholding that ISN'T one of these recognized leaves (e.g. dn_bottleneck.py's
    skip_pad='mvau' identity-pad conv), which must still get a 'skip_quant' role or its FIFO is never
    matched by step_force_fifo_depths_from_milp and silently stays at FINN's shallow default depth."""
    claimed = set()
    weight_nodes = [n for n in kernel_model.graph.node if n.op_type in WEIGHT_OP_TYPES]
    for node, logical_name in zip(weight_nodes, logical_names):
        entry, json_key = _resolve_folding_entry(logical_name, per_layer)
        if entry is None or json_key is None:
            continue
        stage, _leaf = _stage_leaf(json_key)
        if stage is not None:
            claimed.add(node.name)
    return claimed


def _find_join_thresholds(kernel_model, logical_names, prev_block=None, claimed_mvau_names=None):
    """{Thresholding node name: (block, kind)} for the residual-join thresholds the MILP prices as
    extra nodes `<block>.skip_quant|residual_add|out_act` (finn_cost_model.md "Residual-join
    thresholds"). Matched structurally around each AddStreams: residual_add = Thresholding
    consuming the add's output; out_act = the Thresholding consuming that; skip_quant = Thresholding
    feeding an add input whose producer is NOT an MVAU/VVAU, OR is an MVAU/VVAU not already claimed
    as a main-path leaf by claimed_mvau_names (dn_bottleneck.py's skip_pad='mvau' identity-pad conv --
    a claimed one is that conv's own thr, handled via _find_following_thresholding instead). Ported from
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
            weight_fed = prod_in is not None and prod_in.op_type in WEIGHT_OP_TYPES
            if weight_fed and (claimed_mvau_names is None or prod_in.name in claimed_mvau_names):
                continue
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


# ----------------------------------------------------------------- FIFO depth bridge (MILP intra/inter_block_fifos)
# Transparent when walking producer/consumer chains to find a FIFO's real neighbours -- DWC is its OWN role ("dwc",
# see _role_name_of), not skipped, same convention as bottleneck_probe_v1/finn_bottleneck_probe_build.py.
_SKIP_FIFO_OPS = ("StreamingFIFO",)


def _real_producer(model, node):
    p = model.find_producer(node.input[0])
    while p is not None and p.op_type.startswith(_SKIP_FIFO_OPS):
        p = model.find_producer(p.input[0])
    return p


def _real_consumer(model, node):
    c = model.find_consumer(node.output[0])
    while c is not None and c.op_type.startswith(_SKIP_FIFO_OPS):
        c = model.find_consumer(c.output[0])
    return c


# Same as above but ALSO transparent through a real DWC node -- used only as a last-resort fallback when a
# FIFO is directly adjacent to a real DWC and net_fold.py modeled this edge as a single direct hop (no
# explicit 'dwc' role at all for this particular pair), so the only way to find a matching 'wanted' entry is
# to walk past the DWC to the true far-side role on both sides of it (see step_force_fifo_depths_from_milp).
_SKIP_FIFO_DWC_OPS = _SKIP_FIFO_OPS + ("StreamingDataWidthConverter",)


def _real_producer_skip_dwc(model, node):
    p = model.find_producer(node.input[0])
    while p is not None and p.op_type.startswith(_SKIP_FIFO_DWC_OPS):
        p = model.find_producer(p.input[0])
    return p


def _real_consumer_skip_dwc(model, node):
    c = model.find_consumer(node.output[0])
    while c is not None and c.op_type.startswith(_SKIP_FIFO_DWC_OPS):
        c = model.find_consumer(c.output[0])
    return c


def _role_name_of(node, role_of_node: dict):
    if node is None:
        return None
    if node.op_type.startswith("StreamingDataWidthConverter"):
        return "dwc"
    return role_of_node.get(node.name)


# per_layer-key leaf suffix (as built by MILP/analytical/net_fold.py's run_block/put calls) -> (mvau role, thr role)
_LEAF_ROLES = {
    "reduce.0": ("mvau_r", "thr_r"), "conv": ("mvau_m", "thr_m"), "conv.0": ("mvau_m", "thr_m"),
    "expand.0": ("mvau_e", "thr_e"), "main_proj.0": ("mvau_p", "thr_p"), "up.0": ("mvau_u", "thr_u"),
}
# the leaf whose MVAU is fed directly by the block's own Dup/DuplicateStreams, per block kind
_ENTRY_LEAF_BY_KIND = {"reg": "reduce.0", "dn": "reduce.0", "up": "main_proj.0", "init": "conv"}


def _block_kind_of_stage(stage: str) -> str:
    if stage == "initial":
        return "init"
    if stage == "final":
        return "final"
    if stage.startswith("down"):
        return "dn"
    if stage.startswith("up"):
        return "up"
    return "reg"


def _stage_leaf(json_key: str):
    """json_key ('stage3.4.reduce.0', 'initial.conv', 'final') -> (stage, leaf), e.g. ('stage3.4', 'reduce.0')."""
    if json_key == "final":
        return "final", "final"
    for leaf in _LEAF_ROLES:
        suffix = f".{leaf}"
        if json_key.endswith(suffix):
            return json_key[: -len(suffix)], leaf
    return None, None


def _find_init_block_extra_nodes(kernel_model, stage, dup_node, conv_path_head):
    """'init'-kind block's non-weight-bearing extra nodes (thr_in, thr_m, maxpool, concat, thr_act),
    resolved by walking out from the already-found Dup node per int_bottleneck.py's topology:
        Thr_in -> Dup -+-> [conv_path_head: FMPad or SWG, already resolved by the caller] -> ... -+
                       +-> Thr_m -> MaxPool -------------------------------------------------------+-> Concat -> Thr_act
    Also registers net_fold.py's own MILP extra-node names (input_quant/act/pool_quant, see net_fold.py's
    'init' kind xf[...] entries) as aliases for the same real nodes -- inter_block_fifos is keyed by those,
    not by int_bottleneck.py's _ROLE_OF names (which intra_block_fifos uses)."""
    by_name: dict[str, str] = {}
    thr_in = _real_producer(kernel_model, dup_node)
    if thr_in is not None and thr_in.op_type in THRESH_OP_TYPES:
        by_name[f"{stage}.thr_in"] = thr_in.name
        by_name[f"{stage}.input_quant"] = thr_in.name
    pool_head = None
    for out_tensor in dup_node.output:
        c = kernel_model.find_consumer(out_tensor)
        while c is not None and c.op_type.startswith("StreamingFIFO"):
            c = kernel_model.find_consumer(c.output[0])
        if c is not None and (conv_path_head is None or c.name != conv_path_head.name):
            pool_head = c
    maxpool = None
    if pool_head is not None and pool_head.op_type in THRESH_OP_TYPES:
        by_name[f"{stage}.thr_m"] = pool_head.name
        by_name[f"{stage}.pool_quant"] = pool_head.name
        maxpool = _real_consumer(kernel_model, pool_head)
    elif pool_head is not None and pool_head.op_type.startswith("StreamingMaxPool"):
        maxpool = pool_head
    concat = None
    if maxpool is not None and maxpool.op_type.startswith("StreamingMaxPool"):
        by_name[f"{stage}.maxpool"] = maxpool.name
        concat = _real_consumer(kernel_model, maxpool)
    if concat is not None and concat.op_type.startswith("StreamingConcat"):
        by_name[f"{stage}.concat"] = concat.name
        thr_act = _real_consumer(kernel_model, concat)
        if thr_act is not None and thr_act.op_type in THRESH_OP_TYPES:
            by_name[f"{stage}.thr_act"] = thr_act.name
    return by_name


def _find_dn_block_via_skip(kernel_model, skip_mvau_node):
    """'dn'-kind block's per_layer leaf 'reduce.0' is actually the MILP's SKIP-branch conv (role 'mvau_s'),
    fed DIRECTLY by its own StreamingMaxPool (no SWU -- a true 1x1), itself fed directly by Dup. The MAIN
    chain's entry conv ('shortcut_proj', role 'mvau_r') sits on Dup's OTHER branch behind its own SWU
    (swg_r) and has NO per_layer/logical_name entry at all (never claimed by the main per-leaf loop above).
    Confirmed via direct topology trace (recheck_p1_topology.py), contradicting the naive assumption
    'reduce.0 == MILP's mvau_r' that 'reg' kind's (truly-1x1, no-SWU, main-chain-entry) reduce.0 would
    suggest:
        Dup -+-> MaxPool -> MVAU_s(='reduce.0' per_layer leaf) -> Thr_s --(skip FIFO)--> Add
             +-> SWG_r -> MVAU_r(='shortcut_proj', UNCLAIMED) -> Thr_r -> FMPad -> SWG_m -> MVAU_m(='conv.0') -> ...
    Returns (maxpool_node|None, dup_node|None, swg_r_node|None, shortcut_mvau_node|None)."""
    maxpool = _real_producer_skip_dwc(kernel_model, skip_mvau_node)
    if maxpool is None or not maxpool.op_type.startswith("StreamingMaxPool"):
        return None, None, None, None
    dup = _real_producer_skip_dwc(kernel_model, maxpool)
    if dup is None or not dup.op_type.startswith("DuplicateStreams"):
        return maxpool, None, None, None
    swg_r = shortcut_mvau = None
    for out_tensor in dup.output:
        c = kernel_model.find_consumer(out_tensor)
        while c is not None and c.op_type.startswith(_SKIP_FIFO_DWC_OPS):
            c = kernel_model.find_consumer(c.output[0])
        if c is not None and c.name != maxpool.name and c.op_type in SWU_OP_TYPES:
            swg_r = c
            shortcut_mvau = _real_consumer_skip_dwc(kernel_model, swg_r)
    return maxpool, dup, swg_r, shortcut_mvau


def _find_up_block_upnn(kernel_model, thr_p_node):
    """'up'-kind block's UpsampleNearestNeighbour, immediately downstream of main_proj's own Thr_p
    (Dup -> MVAU_p -> Thr_p -> UpNN -> ...; per up_bottleneck.py's topology docstring)."""
    c = _real_consumer(kernel_model, thr_p_node)
    if c is not None and c.op_type.startswith("UpsampleNearestNeighbour"):
        return c
    return None


def build_partition_role_nodes(kernel_model, logical_names, per_layer):
    """Stage-qualified role name (e.g. 'stage3.4.mvau_r', 'stage3.4.thr_r', 'stage3.4.dup') -> real FINN node
    name, for every weight-bearing layer in this partition. Covers each block's entry/mid/exit MVAU and their
    OWN Thresholding (mvau_r/thr_r, mvau_m/thr_m, mvau_e/thr_e, mvau_p/thr_p, mvau_u/thr_u, mvau_c/thr_c,
    mvau_f), the dense-conv's FMPadding/SWU (fmpad/swg_m, via _find_dense_swu_fmpad, resolved for 'init' kind
    too; 'up' kind's own 2x2-lowered conv gets fmpadpix/fmpad_u/swg_u via _find_up_block_fmpad_chain, which
    unlike the dense case has TWO chained FMPadding nodes to resolve), each block's Dup (feeding its
    entry MVAU), 'dn' kind's own skip-branch MaxPool/SWG_r/shortcut-conv (mvau_r/thr_r, UNCLAIMED by the
    main per-leaf loop -- see _find_dn_block_via_skip for why 'reduce.0' is actually the MILP's 'mvau_s'
    skip-conv, not 'mvau_r'), 'up' kind's own UpsampleNearestNeighbour (_find_up_block_upnn), and -- for
    'init' kind only -- its own thr_in/thr_m/maxpool/concat/thr_act (_find_init_block_extra_nodes). Does NOT
    cover add/skip_quant/residual_add/out_act -- see _find_block_join_nodes."""
    weight_nodes = [n for n in kernel_model.graph.node if n.op_type in WEIGHT_OP_TYPES]
    by_name: dict[str, str] = {}
    for node, logical_name in zip(weight_nodes, logical_names):
        entry, json_key = _resolve_folding_entry(logical_name, per_layer)
        if entry is None or json_key is None:
            continue
        stage, leaf = _stage_leaf(json_key)
        if stage is None:
            continue
        if leaf == "final":
            by_name[f"{stage}.mvau_f"] = node.name
            continue
        kind = _block_kind_of_stage(stage)
        mvau_role, thr_role = _LEAF_ROLES[leaf]
        if leaf == "conv" and stage == "initial":
            mvau_role, thr_role = "mvau_c", "thr_c"
        elif leaf == "reduce.0" and kind == "dn":
            # see _find_dn_block_via_skip: dn_bottleneck.py's 'reduce.0' is the MILP's SKIP-branch conv,
            # not the main-chain entry _LEAF_ROLES assumes (that's 'reg' kind's shape, not 'dn' kind's).
            mvau_role, thr_role = "mvau_s", "thr_s"
        by_name[f"{stage}.{mvau_role}"] = node.name
        thr = _find_following_thresholding(kernel_model, node)
        if thr is not None:
            by_name[f"{stage}.{thr_role}"] = thr.name
        fmpad_node = swu_node = None
        if leaf in ("conv", "conv.0"):
            fmpad_node, swu_node = _find_dense_swu_fmpad(kernel_model, node)
            if swu_node is not None:
                # net_fold.py role names for this SWU differ by block kind: dn_bottleneck.py -> 'swg_m',
                # bottleneck.py (reg kind) -> plain 'swg' (its _ROLE_OF maps FINN's own 'SWG_m' var name to
                # role string 'swg'). Register both so either producer looks it up.
                by_name[f"{stage}.swg_m"] = swu_node.name
                by_name[f"{stage}.swg"] = swu_node.name
            if fmpad_node is not None:
                by_name[f"{stage}.fmpad"] = fmpad_node.name
        elif leaf == "up.0":
            # up_bottleneck.py's 2x2-lowered transposed conv: FMPadPix -> FMPad_u -> SWG_u -> MVAU_u -- TWO
            # chained FMPadding nodes (roles 'fmpadpix'/'fmpad_u'), unlike every other conv's single
            # FMPadding -- see _find_up_block_fmpad_chain's docstring for why _find_dense_swu_fmpad alone
            # under-finds/mislabels this case.
            fmpadpix_node, fmpad_u_node, swu_node = _find_up_block_fmpad_chain(kernel_model, node)
            fmpad_node = fmpad_u_node
            if swu_node is not None:
                by_name[f"{stage}.swg_u"] = swu_node.name
            if fmpad_u_node is not None:
                by_name[f"{stage}.fmpad_u"] = fmpad_u_node.name
            if fmpadpix_node is not None:
                by_name[f"{stage}.fmpadpix"] = fmpadpix_node.name
        if leaf == _ENTRY_LEAF_BY_KIND.get(kind):
            if kind == "dn":
                # 'reduce.0' (this node, role 'mvau_s') is fed directly by MaxPool, itself fed directly by
                # Dup -- walk back through those (no FMPad/SWU on THIS branch) to find Dup, then across to
                # the other, unclaimed main-chain-entry branch (swg_r -> mvau_r/thr_r).
                maxpool, dup, swg_r, shortcut_mvau = _find_dn_block_via_skip(kernel_model, node)
                if maxpool is not None:
                    by_name[f"{stage}.maxpool"] = maxpool.name
                if dup is not None:
                    by_name[f"{stage}.dup"] = dup.name
                if swg_r is not None:
                    by_name[f"{stage}.swg_r"] = swg_r.name
                if shortcut_mvau is not None and shortcut_mvau.op_type in WEIGHT_OP_TYPES:
                    by_name[f"{stage}.mvau_r"] = shortcut_mvau.name
                    thr_r = _find_following_thresholding(kernel_model, shortcut_mvau)
                    if thr_r is not None:
                        by_name[f"{stage}.thr_r"] = thr_r.name
            else:
                # 'init' kind's entry leaf ('conv') has FMPad/SWG/DWC between Dup and the MVAU itself; every
                # other (non-'dn') kind's entry MVAU is fed directly by Dup.
                dup = _real_producer(kernel_model, fmpad_node or swu_node or node)
                if dup is not None and dup.op_type.startswith("DuplicateStreams"):
                    by_name[f"{stage}.dup"] = dup.name
                    if kind == "init":
                        by_name.update(_find_init_block_extra_nodes(kernel_model, stage, dup, fmpad_node or swu_node))
        if leaf == "main_proj.0" and thr is not None:
            upnn = _find_up_block_upnn(kernel_model, thr)
            if upnn is not None:
                by_name[f"{stage}.upnn"] = upnn.name
    return by_name


def _find_block_join_nodes(kernel_model, logical_names, prev_block=None, claimed_mvau_names=None):
    """role -> node bridge for the residual-join nodes the MILP's inter/intra_block_fifos reference as
    '<block>.add' / '<block>.skip_quant' (='<block>.thr_s') / '<block>.residual_add' / '<block>.out_act'
    (aliased to '<block>.thr_out' -- the block's externally-visible output, what the NEXT block's Dup is fed
    by; net_fold.py's per-block sim calls this one conceptual node 'Thr_out'). Reuses _find_join_thresholds
    (proven) for residual_add/out_act/skip_quant; adds the AddStreams node itself via the same
    adds/expand.0-blocks zip that function uses internally."""
    by_name: dict[str, str] = {}
    joins = _find_join_thresholds(kernel_model, logical_names, prev_block, claimed_mvau_names)
    for node_name, (block, kind) in joins.items():
        by_name[f"{block}.{kind}"] = node_name
        if kind == "skip_quant":
            by_name[f"{block}.thr_s"] = node_name
        if kind == "out_act":
            by_name[f"{block}.thr_out"] = node_name
            by_name[f"{block}.out_act"] = node_name  # net_fold.py's intra/inter_block_fifos role name for this node
    for key in list(by_name):
        # thr_out/out_act fallback: hardware/finn_compose_thresholds.py's step_compose_consecutive_thresholds
        # (wired into the real preamble) merges the real residual_add + out_act Thresholding nodes into ONE
        # node, so _find_join_thresholds above never finds a SEPARATE out_act node and 'kind == "out_act"'
        # above never fires -- without this fallback 'down1.out_act' (etc.) is simply absent from role_by_name,
        # which net_fold.py's inter_block_fifos producer field always uses for the dup-transition edge
        # ('<block>.out_act' -> '<next_block>.dup'), silently dropping every one of those FIFOs to
        # fully-unresolved (NOT the producer-only inter-block convention -- it never even gets that far).
        # net_fold.py's own extra_entry() calls residual_add a zero-cost placeholder "merged into" out_act
        # (merged_into=...+'out_act'), so aliasing BOTH names to the one real merged node matches that
        # convention exactly.
        if key.endswith(".residual_add"):
            block = key[: -len(".residual_add")]
            by_name.setdefault(f"{block}.thr_out", by_name[key])
            by_name.setdefault(f"{block}.out_act", by_name[key])
    adds = [n for n in kernel_model.graph.node if n.op_type.startswith("AddStreams")]
    blocks = [ln[: -len(".expand.0")] for ln in logical_names if ln.endswith(".expand.0")]
    if len(adds) == len(blocks):
        for add, block in zip(adds, blocks):
            by_name[f"{block}.add"] = add.name
    return by_name


def step_set_fifo_depths_fixed2(model, cfg):
    """Default FIFO-depth strategy for S12-dense builds -- skips FINN's own auto-sizing entirely (both
    'characterize' and 'largefifo_rtlsim' run real HLS-rtlsim/Verilator cosimulation, which is what makes
    builds slow and also what produced the huge, MILP-unaware autosized depths seen on un-bridged edges,
    e.g. a near-full-frame depth on the edge into LabelSelect). Instead: insert a real StreamingFIFO node on
    EVERY HW-to-HW edge (create_shallow_fifos=True) at FINN's structural default depth (2, from InsertFIFO's
    max(outFIFODepths, inFIFODepths) node-attr defaults), and deliberately do NOT call RemoveShallowFIFOs --
    every inserted FIFO must stay a real node so step_force_fifo_depths_from_milp (called right after this)
    can see, override, and report EVERY one of them, matched or not."""
    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO(create_shallow_fifos=True, vivado_ram_style=cfg.large_fifo_mem_style))
    model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model


def step_force_fifo_depths_from_milp(model, fifo_plan: dict, min_depth: int = 2, skip_scale: float = 1.0):
    """After FINN's own (autosized) step_set_fifo_depths, overwrite StreamingFIFO* node depths with the
    MILP/analytical per-block-simulated ones (fifo_plan's 'wanted', keyed '<producer_role>=><consumer_role>'),
    matched by role identity via fifo_plan's 'role_of_node' (node name -> role string) + _real_producer/
    _real_consumer. Edges with no match keep FINN's autosized depth (safety net -- same pattern as
    hardware/builds/bottleneck_probe_v1/finn_bottleneck_probe_build.py's step_force_fifo_depths, generalized
    from one isolated block to a whole partition's worth of block instances). Every report entry (matched or
    not) also carries the REAL producer/consumer node name + op_type, so unmatched edges are directly
    actionable without a separate describe-the-graph pass."""
    role_of_node = fifo_plan["role_of_node"]
    wanted = {tuple(k.split("=>", 1)): v for k, v in fifo_plan["wanted"].items()}
    wanted_by_producer = fifo_plan.get("wanted_by_producer", {})
    n_total = n_forced = n_forced_virtual_dwc = n_forced_producer_only = n_forced_dwc_bridge = 0
    report = []
    for n in model.graph.node:
        if not n.op_type.startswith("StreamingFIFO"):
            continue
        n_total += 1
        inst = getCustomOp(n)
        prod, cons = _real_producer(model, n), _real_consumer(model, n)
        pr_role, cn_role = _role_name_of(prod, role_of_node), _role_name_of(cons, role_of_node)
        # a DWC node has no stage of its own -- borrow it from whichever side already resolved to a real role
        if pr_role == "dwc" and cn_role and cn_role != "dwc" and "." in cn_role:
            pr_role = f"{cn_role.rsplit('.', 1)[0]}.dwc"
        elif cn_role == "dwc" and pr_role and pr_role != "dwc" and "." in pr_role:
            cn_role = f"{pr_role.rsplit('.', 1)[0]}.dwc"
        stock = inst.get_nodeattr("depth")
        entry = dict(
            fifo=n.name,
            producer_node=f"{prod.name}({prod.op_type})" if prod is not None else None,
            consumer_node=f"{cons.name}({cons.op_type})" if cons is not None else None,
            producer_role=pr_role, consumer_role=cn_role, stock_depth=stock, forced_depth=None,
        )
        f = wanted.get((pr_role, cn_role)) if (pr_role and cn_role) else None
        producer_only = False
        if f is None and cons is None and pr_role:
            # partition's own last FIFO -- its real consumer lives in a not-yet-built partition, so there's no
            # node here to pair against. Convention: an inter-block FIFO belongs to the block that emits it, so
            # match by producer role alone instead of leaving this on FINN's autosized depth.
            f = wanted_by_producer.get(pr_role)
            producer_only = f is not None
        dwc_bridge = False
        if f is None and (
            (prod is not None and prod.op_type.startswith("StreamingDataWidthConverter"))
            or (cons is not None and cons.op_type.startswith("StreamingDataWidthConverter"))
        ):
            # net_fold.py doesn't consistently model an explicit 'dwc' role on every edge a real DWC can land
            # on -- some edges are modeled as one direct pr_role->cn_role hop with no dwc stage at all, even
            # though FINN's real InsertDWC did materialize a DWC there. Walk past the real DWC node (on
            # whichever side it sits) to the true far-side role and retry a direct lookup; the same modeled
            # depth is then forced onto BOTH real FIFOs flanking that DWC.
            true_pr_role, true_cn_role = pr_role, cn_role
            if prod is not None and prod.op_type.startswith("StreamingDataWidthConverter"):
                true_pr_role = _role_name_of(_real_producer_skip_dwc(model, n), role_of_node)
            if cons is not None and cons.op_type.startswith("StreamingDataWidthConverter"):
                true_cn_role = _role_name_of(_real_consumer_skip_dwc(model, n), role_of_node)
            if true_pr_role and true_cn_role and true_pr_role != "dwc" and true_cn_role != "dwc":
                f = wanted.get((true_pr_role, true_cn_role))
                dwc_bridge = f is not None
        virtual_dwc = False
        if f is None and pr_role and cn_role and not pr_role.endswith(".dwc") and not cn_role.endswith(".dwc"):
            # net_fold.py's per-block sim always models an explicit DWC stage between certain role pairs; FINN's
            # real InsertDWC only materializes one when widths actually differ, so the real edge here can be a
            # direct pr_role->cn_role hop with no 'dwc' node in between. Collapse the sim's two-edge chain
            # (pr_role->dwc, dwc->cn_role) into this one real edge, using the larger of the two depths.
            stage = pr_role.rsplit(".", 1)[0]
            leg1, leg2 = wanted.get((pr_role, f"{stage}.dwc")), wanted.get((f"{stage}.dwc", cn_role))
            if leg1 is not None and leg2 is not None:
                f = leg1 if leg1["depth"] >= leg2["depth"] else leg2
                virtual_dwc = True
        if f is not None:
            depth = max(min_depth, int(round(f["depth"] * (skip_scale if f.get("is_skip") else 1.0))))
            inst.set_nodeattr("depth", depth)
            entry["forced_depth"] = depth
            if inst.get_nodeattr("impl_style") != "rtl":
                inst.set_nodeattr("impl_style", "rtl")
                reset_implementation(inst)
                entry["impl_style_forced_to_rtl"] = True
            if virtual_dwc:
                entry["forced_via_virtual_dwc_collapse"] = True
                n_forced_virtual_dwc += 1
            if producer_only:
                entry["forced_via_producer_only_inter_block"] = True
                n_forced_producer_only += 1
            if dwc_bridge:
                entry["forced_via_dwc_bridge"] = True
                n_forced_dwc_bridge += 1
            n_forced += 1
        report.append(entry)
    print(f"[force fifo depths from MILP] forced {n_forced}/{n_total} StreamingFIFO node(s) "
          f"({n_forced_virtual_dwc} via virtual-DWC collapse, {n_forced_dwc_bridge} via walk-past-DWC bridge, "
          f"{n_forced_producer_only} via producer-only inter-block "
          f"convention); {n_total - n_forced} left unassigned (kept stock_depth)")
    unassigned = [e for e in report if e["forced_depth"] is None]
    if unassigned:
        print(f"[force fifo depths from MILP] {len(unassigned)} unassigned FIFO(s) -- producer -> consumer "
              f"(stock_depth):")
        for e in unassigned:
            print(f"    {e['fifo']:28s} {str(e['producer_node']):40s} -> {str(e['consumer_node']):40s} "
                  f"stock_depth={e['stock_depth']}")
    return model, report


def build_partition_folding_config(partition_model_fn, sdp_node_name, logical_names, per_layer, cfg, tag="",
                                    extra_nodes=None, conv_order_file=None,
                                    inter_block_fifos=None, intra_block_fifos=None):
    """MILP per_layer (pe/simd/simd_swu/thr_pe/thr_ram_style) -> FINN folding
    config keyed by the node names step_apply_folding_config will see.
    extra_nodes (the MILP folding json's own "extra_nodes" dict) + conv_order_file (the FULL,
    cross-partition conv_order.json) are optional: when both given, also bridges the residual-join
    Thresholding nodes' (skip_quant/residual_add/out_act) depth_trigger_bram from their own
    ram_style -- without them, join thresholds are left on Vivado auto placement.
    inter_block_fifos/intra_block_fifos (the MILP folding json's own top-level lists) are also optional:
    when either is given, this ALSO resolves a FIFO-forcing plan (same role-identity bridge, reusing this
    function's own kernel_model) for step_force_fifo_depths_from_milp -- returned as a second value instead
    of None. Returns (folding_config, fifo_plan_or_None)."""
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
            _, up0_leaf = _stage_leaf(json_key)
            if up0_leaf == "up.0":
                # up_bottleneck.py's 2x2-lowered conv has TWO chained FMPadding nodes (FMPadPix, FMPad_u)
                # before its SWG -- _find_dense_swu_fmpad only finds the nearer FMPad_u, so FMPadPix's own
                # SIMD would otherwise silently stay at FINN's default/auto instead of matching the chain.
                fmpadpix_node, fmpad_node, swu_node = _find_up_block_fmpad_chain(kernel_model, node)
            else:
                fmpad_node, swu_node = _find_dense_swu_fmpad(kernel_model, node)
                fmpadpix_node = None
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
                if fmpadpix_node is not None:
                    folding_config[fmpadpix_node.name] = {"SIMD": simd_swu}

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

    claimed_mvau_names = _claimed_mvau_names(kernel_model, logical_names, per_layer)

    # residual-join thresholds: memory placement only (their MILP PE is not applied here -- they
    # stay at FINN's own PE, same as the original partition2_wm bridge this was ported from).
    n_join = 0
    if extra_nodes is not None and conv_order_file is not None:
        prev_block = _previous_block_name(conv_order_file, logical_names[0]) if logical_names else None
        joins = _find_join_thresholds(kernel_model, logical_names, prev_block, claimed_mvau_names)
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

    fifo_plan = None
    if inter_block_fifos is not None or intra_block_fifos is not None:
        role_by_name = build_partition_role_nodes(kernel_model, logical_names, per_layer)
        prev_block = _previous_block_name(conv_order_file, logical_names[0]) if (conv_order_file and logical_names) else None
        role_by_name.update(_find_block_join_nodes(kernel_model, logical_names, prev_block, claimed_mvau_names))
        # Several real nodes are registered under more than one role string (int_bottleneck.py's _ROLE_OF name
        # AND net_fold.py's own MILP xf name for the SAME node, e.g. 'initial.thr_act' / 'initial.act' both ->
        # the init block's output Thresholding) -- intra_block_fifos and inter_block_fifos don't consistently
        # agree on which alias they use for a given node, so role_of_node (used at runtime) can only surface
        # ONE alias per node, but 'wanted'/'wanted_by_producer' below are populated under EVERY alias so a
        # match succeeds no matter which alias runtime happens to resolve to.
        aliases_of_node: dict = {}
        for role, node_name in role_by_name.items():
            aliases_of_node.setdefault(node_name, set()).add(role)
        role_of_node = {v: k for k, v in role_by_name.items()}

        def _resolved_role(role):
            # usually role IS the bridge's role string already; net_fold.py's extra_nodes also carries each
            # node's own "canonical_role" ('<consuming stage>.<kind>') for the rare case (dup) where its MILP
            # name disagrees with that convention (producer-qualified instead of consumer-qualified).
            if role in role_by_name:
                return role
            if extra_nodes is not None:
                canonical = (extra_nodes.get(role) or {}).get("canonical_role")
                if canonical in role_by_name:
                    return canonical
            return None

        def _aliases(role):
            resolved = _resolved_role(role)
            node_name = role_by_name.get(resolved) if resolved is not None else None
            return aliases_of_node.get(node_name, {role}) if node_name is not None else {role}

        def _dwc_resolvable(role):
            # a DWC node has no fixed identity in role_by_name (a stage can have several real DWC instances) --
            # it's resolved dynamically at runtime by borrowing the stage from whichever side is a real role
            # (see step_force_fifo_depths_from_milp), so any '<stage>.dwc'-suffixed role is treated as
            # bridge-time resolvable without needing a literal role_by_name entry.
            return isinstance(role, str) and role.endswith(".dwc")

        def _role_ok(role):
            return _resolved_role(role) is not None or _dwc_resolvable(role)

        wanted, wanted_by_producer, unresolved = {}, {}, []
        for f in (intra_block_fifos or []):
            pr, cn = f["producer"], f["consumer"]
            if _role_ok(pr) and _role_ok(cn):
                for pr_alias in _aliases(pr):
                    for cn_alias in _aliases(cn):
                        wanted[f"{pr_alias}=>{cn_alias}"] = f
            else:
                unresolved.append([pr, cn])
        for f in (inter_block_fifos or []):
            pr, cn = f["producer"], f["consumer"]
            pr_ok, cn_ok = _role_ok(pr), _role_ok(cn)
            if pr_ok and cn_ok:
                for pr_alias in _aliases(pr):
                    for cn_alias in _aliases(cn):
                        wanted[f"{pr_alias}=>{cn_alias}"] = f
            elif pr_ok:
                # convention: an inter-block FIFO belongs to the block that emits it, so it's still resolvable
                # by producer role alone when the consumer's block lives in a different (not yet built) partition.
                for pr_alias in _aliases(pr):
                    wanted_by_producer[pr_alias] = f
            else:
                unresolved.append([pr, cn])
        all_fifos = (inter_block_fifos or []) + (intra_block_fifos or [])
        print(f"{log} FIFO role bridge: {len(role_by_name)} role(s) resolved; {len(wanted)}/{len(all_fifos)} FIFO "
              f"spec(s) matched to a real node pair, {len(wanted_by_producer)} more matchable via producer-only "
              f"inter-block convention ({len(unresolved)} fully unresolved)")
        fifo_plan = {"role_of_node": role_of_node, "wanted": wanted, "wanted_by_producer": wanted_by_producer,
                     "unresolved": unresolved}

    return folding_config, fifo_plan
