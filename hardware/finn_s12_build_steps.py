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
# Forces thresholding.sv's RAM_STYLE to "distributed" for every real depth.
THRESH_DISTRIBUTED_BRAM_TRIGGER = 999999
PARTITION_RANGE_ORDER = [
    "down1_start", "down2_start", "q2_start", "q3_start", "q4_start", "up4_start", "up5_start",
]


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


def load_partition_logical_names(preamble_dir, conv_order_file, n_partitions=8):
    """{partition_idx: (conv_logical_names, pool_logical_names)} via positional
    match of conv_order.json against the pre-partition graph. A forked MatMul
    duplicated by step_dedup_forked_matmul_before_threshold (same weight
    tensor) inherits the first copy's logical name."""
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
        wt = _weight_tensor(full_model.graph.node[node_idx])
        if wt is not None and wt in tensor_to_entry:
            node_idx_to_entry[node_idx] = tensor_to_entry[wt]
            continue
        if pos >= len(all_names):
            raise RuntimeError(f"ran out of conv_order.json entries at node_idx={node_idx} -- do not proceed.")
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


def build_partition_folding_config(partition_model_fn, sdp_node_name, logical_names, per_layer, cfg, tag=""):
    """MILP per_layer (pe/simd/simd_swu/thr_pe/thr_ram_style) -> FINN folding
    config keyed by the node names step_apply_folding_config will see."""
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
                swu_cfg = {"SIMD": simd_swu}
                if swu_node.op_type == "ConvolutionInputGenerator_rtl":
                    # Mirrors finn_cost_model.conv_cost_pe_simd: parallel_window iff MVAU SIMD > cin.
                    ifm_ch = getCustomOp(swu_node).get_nodeattr("IFMChannels")
                    parallel_window = 1 if simd > ifm_ch else 0
                    swu_cfg["parallel_window"] = parallel_window
                    expected = ifm_ch if parallel_window else math.gcd(simd, ifm_ch)
                    if expected != simd_swu:
                        print(f"{log} WARNING {swu_node.name}: MILP simd_swu={simd_swu} but cost-model rule gives {expected}")
                folding_config[swu_node.name] = swu_cfg
                print(f"{log} {swu_node.name:30s} {swu_node.op_type:12s} (SWU) {swu_cfg}")
                if fmpad_node is not None:
                    folding_config[fmpad_node.name] = {"SIMD": simd_swu}

        thr_pe = compute_entry.get("thr_pe")
        if thr_pe is not None:
            thresh_node = _find_following_thresholding(kernel_model, node)
            if thresh_node is not None:
                thr_config = {"PE": thr_pe}
                if compute_entry.get("thr_ram_style") == "distributed":
                    thr_config["depth_trigger_bram"] = THRESH_DISTRIBUTED_BRAM_TRIGGER
                folding_config[thresh_node.name] = thr_config
                print(f"{log} {thresh_node.name:30s} {thresh_node.op_type:12s} (thr_pe) {thr_config}")

    if unmatched:
        print(f"{log} WARNING: {len(unmatched)} logical names had no folding entry: {unmatched}")
    return folding_config
