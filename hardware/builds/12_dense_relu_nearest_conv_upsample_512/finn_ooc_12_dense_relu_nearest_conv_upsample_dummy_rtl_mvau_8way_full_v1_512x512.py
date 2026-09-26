"""v1 (dummy-weight) 8-way full build for the NEW
12_dense_relu_nearest_conv_upsample architecture (decoder_type=
"nearest_conv_upsample" -- see enet/nnunetv2/nets/LayerQuantEnetFINN.py's
FINNUpsamplingBottleneck docstring and MILP/config_12_dense_relu_
nearest_conv_upsample.py), 512x512 input, adapted directly from the last
known-good reference build,
finn_ooc_12_dense_relu_warmstart150ep_alpha05_trained_rtl_mvau_8way_full_v4_512x512.py
("_v4" -- per explicit instruction). Same channels/bottleneck-depths/
context-pattern/8-way-partitioning/DSP-forcing/BIPOLAR-weight-dtype-fix/
standalone-threshold-aware accumulator widening as _v4; only real
architecture deltas vs _v4:
  - decoder_type="nearest_conv_upsample": up4/up5's main_up is now a bare
    nn.Upsample(mode="nearest") (NO frozen depthwise blur conv at all --
    unlike the old decoder, the real op here already IS plain nearest
    resize, so FINN needs no approximation substitute), followed by a REAL
    LEARNED dense skip_resize_conv (3x3, groups=1) that lowers to a normal
    MVAU via LowerConvsToMatMul -- NOT a VVAU. This architecture therefore
    has 0 depthwise/VVAU nodes anywhere (unlike _v4's up4/up5 blur convs) --
    the VVAU/SWU PE-scale-up fix in build_partition_folding_config is kept
    for parity/safety but should never fire here.
  - MODEL_NAME/CONV_ORDER_FILE point at the DUMMY (fresh random weight)
    export, per explicit instruction to "start the dummy build also when
    ready just to discover any early errors" ahead of the real ft15ep QAT
    checkpoint landing. Swap to the "_trained" export + FOLDING_BLOCK_FILE's
    already-correct alpha=1.0 solve once that checkpoint is exported.
  - FOLDING_BLOCK_FILE points at the DEFINITIVE production MILP solve for
    this architecture (confirmed via the real QAT fine-tuning SLURM job,
    not guessed): alpha=1.0 (pure sensitivity objective), hard caps
    LUT<=50%/BRAM<=20%/DSP<=90%, max-latency<=200ms, max-join-imbalance-
    ratio<=1.1 (binds the up4/up5 skip_resize_conv.0-vs-expand.0 residual
    join).

Three deltas vs _v4's per-partition builder (all per explicit instruction):
  1. Dedicated build directory: _BASE_BUILD_DIR now ends in
     enet/finn_build_tmp/S12_dense_nn_upsample_conv_v1/ (one GenericPartition_N
     subfolder per partition beneath it, same as _v4's own convention) so
     this architecture's build artifacts never collide with or get
     overwritten by any other architecture's build under the same
     finn_build_tmp/ root.
  2. URAM-FIFO allocation (step_allocate_uram_fifos, new): after
     step_set_fifo_depths (FINN's own auto-sizing pass) but BEFORE
     SplitLargeFIFOs, greedily sets ram_style="ultra" on the deepest
     StreamingFIFO_rtl nodes (impl_style=="vivado" only -- impl_style=="rtl"
     Q_srl FIFOs are shallow <=256-deep SRL chains with no BRAM/URAM
     primitive at all) up to a hard 92-block URAM288 budget (ZCU7EV has 96
     URAM288 blocks total; 92 leaves a small margin for anything else that
     might need one). Applied PRE-split specifically so SplitLargeFIFOs'
     own children inherit the parent's ram_style verbatim (confirmed via
     direct FINN source read of set_fifo_depths.py's SplitLargeFIFOs.apply()
     -- `ram_style=ram_style` is copied onto every split child) -- this
     keeps a whole deep FIFO "chain" together in one contiguous URAM
     allocation with zero extra chain/adjacency-detection logic needed.
  3. FIFO-autosize checkpoint save: immediately after
     step_allocate_uram_fifos (i.e. right before SplitLargeFIFOs runs), the
     per-partition kernel_model is saved to its own
     partition_<i>_prefifo_autosize.onnx next to the main dataflow model
     file -- lets FIFO depths/ram_style be inspected without waiting for the
     full OOC synth to finish.
The existing, already-correctly-placed `GiveUniqueNodeNames(prefix)` call
(right before PrepareIP/HLSSynthIP, exactly as in _v4) is kept as-is and
is sufficient for child-IP name collision avoidance -- confirmed the
current production pipeline never combines partitions into one Vivado
project (always per-partition OOC synth only), so the heavier RTL-module-
name-collision monkeypatch documented for an earlier (combined-synthesis)
build family is not needed here.

Per _v4's own convention, OOC synthesis for each partition runs INLINE
inside that partition's own ProcessPoolExecutor worker, immediately after
CreateStitchedIP -- automatically advancing straight to OOC synth with no
separate/deferred script needed, satisfying the "automatically advance to
OOC synth" instruction (this was already _v4's behavior, unchanged here).

Run inside the FINN container (after the dummy_512x512 preamble has
completed):
    docker exec -e HOME=/tmp/home_dir <container> bash -c \\
        'cd /home/thelegendiv/finn/notebooks/enet && \\
         nohup python3 finn_ooc_12_dense_relu_nearest_conv_upsample_dummy_rtl_mvau_8way_full_v1_512x512.py \\
           finn_deployment_outputs/12_dense_relu_nearest_conv_upsample_dummy_rtl_mvau_512x512_preamble_<timestamp> \\
         > /tmp/12_dense_relu_nearest_conv_upsample_dummy_rtl_mvau_8way_full_v1_512x512.log 2>&1 & disown'
"""
import concurrent.futures
import dataclasses
import json
import math
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.datatype import DataType  # noqa: E402
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402
from qonnx.transformation.infer_datatypes import InferDataTypes  # noqa: E402
from qonnx.util.basic import calculate_matvec_accumulator_range, roundup_to_integer_multiple  # noqa: E402

from finn_stage_partition import (  # noqa: E402
    compute_8way_boundaries,
    validate_partition_single_output,
)

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

import finn.builder.build_dataflow as build  # noqa: E402
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP  # noqa: E402
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP  # noqa: E402
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP  # noqa: E402
from finn.transformation.fpgadataflow.set_fifo_depths import SplitLargeFIFOs  # noqa: E402
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext  # noqa: E402
from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth  # noqa: E402
from finn.util.fpgadataflow import is_fpgadataflow_node  # noqa: E402
from finn_partition_build_steps import (  # noqa: E402
    step_create_dataflow_partition_multi,
    step_combine_partitions,
    step_generate_estimate_reports_multi,
    step_measure_rtlsim_performance_multi,
)

# DUMMY (fresh-random-weight) export/conv-order -- swap both to their
# "_trained" siblings once the real ft15ep QAT checkpoint lands and gets
# exported via finn_export_12_dense_relu_nearest_conv_upsample_trained.py.
MODEL_NAME = "quantEnet_12_dense_relu_nearest_conv_upsample_dummy_int8_512x512"
CONV_ORDER_FILE = os.path.join(base.ENET_DIR, "quantEnet_12_dense_relu_nearest_conv_upsample_dummy_int8_conv_order.json")
# DEFINITIVE production MILP solve for this architecture (confirmed via the
# real QAT SLURM job, not guessed): alpha=1.0, hard LUT/BRAM/DSP fractions
# 50%/20%/90%, max-latency 200ms, max-join-imbalance-ratio 1.1.
FOLDING_BLOCK_FILE = os.path.join(
    base.ENET_DIR,
    "layer_bits_folding_12_dense_relu_nearest_conv_upsample_joint_alpha1.0_candidatebits468_forcedsp_lut50_bram20_dsp90_maxlat200ms_joinbalance1.1.json",
)
WEIGHT_OP_TYPES = ("MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl")
VVAU_OP_TYPES = ("VVAU_hls", "VVAU_rtl")
THRESH_OP_TYPES = ("Thresholding_hls", "Thresholding_rtl")
# Real FINN node types for the sliding-window unit and its preceding padding
# block, feeding a VVAU. UNLIKE the old "upsample_conv" decoder (whose
# up4/up5 frozen bilinear-blur substitute was a real depthwise VVAU), this
# architecture's decoder_type="nearest_conv_upsample" up4/up5 main_up is now
# a bare nn.Upsample (no conv at all) followed by a DENSE skip_resize_conv
# (groups=1, lowers to a normal MVAU via LowerConvsToMatMul) -- so this
# architecture has 0 depthwise/VVAU nodes anywhere (all-dense, non-
# separable dilated context, no depthwise decoder conv either). The VVAU/SWU
# PE-scale-up fix below is kept for parity/safety with the _v4 bridge code
# it was copied from, but should never fire for this architecture.
SWU_OP_TYPES = ("ConvolutionInputGenerator_hls", "ConvolutionInputGenerator_rtl")
FMPAD_OP_TYPES = ("FMPadding_hls", "FMPadding_rtl", "FMPadding_Pixel")

# Sentinel forcing thresholding.sv's RAM_STYLE ternary to "distributed" for
# every real DEPTH (all real per-stage depths here are << this value, and
# depth_trigger_uram is left at 0, so the "ultra" branch never triggers).
THRESH_DISTRIBUTED_BRAM_TRIGGER = 999999

PARTITION_RANGE_ORDER = [
    "down1_start", "down2_start", "q2_start", "q3_start", "q4_start", "up4_start", "up5_start",
]

# Hardcoded (not read from os.environ) so every worker derives the same base
# regardless of call order -- each partition then gets its own GenericPartition_N
# subfolder below, keeping build artifacts from mixing across partitions.
# Dedicated per-architecture subfolder (per explicit instruction) so this
# build's artifacts never collide with any other architecture's under the
# same finn_build_tmp/ root.
_BASE_BUILD_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_nn_upsample_conv_v1"

# ZCU7EV has 96 URAM288 blocks total (4096-deep x 72b each); 92 leaves a
# small margin. Only StreamingFIFO_rtl nodes with impl_style=="vivado" are
# eligible -- impl_style=="rtl" (Q_srl) FIFOs are shallow SRL chains with no
# BRAM/URAM primitive at all.
URAM_BUDGET_BLOCKS = 92
URAM_DEPTH_PER_BLOCK = 4096
URAM_WIDTH_PER_BLOCK = 72


def partition_node_index_range(partition_idx, boundaries):
    edges = [0] + [boundaries[k] for k in PARTITION_RANGE_ORDER] + [None]
    return edges[partition_idx], edges[partition_idx + 1]


def load_all_partition_logical_names(preamble_dir):
    # rtl_mvau preamble's converted checkpoint (standalone Thresholding,
    # noActivation=1 everywhere) -- NOT the standard step_enet_convert_to_hw.onnx.
    pre_partition_ckpt = os.path.join(preamble_dir, "intermediate_models", "step_enet_convert_to_hw_rtl_mvau.onnx")
    full_model = ModelWrapper(pre_partition_ckpt)

    boundaries = compute_8way_boundaries(full_model)
    print(f"[bridge] 8-way boundaries: {boundaries}")

    with open(CONV_ORDER_FILE) as f:
        all_names = json.load(f)

    weight_like_idx = [
        idx for idx, node in enumerate(full_model.graph.node)
        if node.op_type in ("MatrixVectorActivation", "MVAU", "VVAU") or "MaxPool" in node.op_type
    ]

    # step_dedup_forked_matmul_before_threshold (finn_enet_build_decomposed_
    # prelu.py) deliberately DUPLICATES a MatMul node -- new node, SAME
    # weight initializer tensor -- whenever its output has multiple
    # MultiThreshold consumers (a workaround for a real FINN
    # InferQuantizedMatrixVectorActivation bug, see that step's docstring).
    # Both copies lower to separate MVAU nodes sharing one weight tensor, so
    # the real graph can have MORE weight-like nodes than conv_order.json
    # has logical names. Detect this by weight-tensor-name reuse: a
    # duplicate consumes NO new conv_order.json entry, instead inheriting
    # the logical name already assigned to the first node that used that
    # same tensor.
    def _weight_tensor(node):
        for inp in node.input:
            if full_model.get_initializer(inp) is not None:
                return inp
        return None

    tensor_to_entry = {}
    pos = 0
    node_idx_to_entry = {}
    for node_idx in weight_like_idx:
        node = full_model.graph.node[node_idx]
        wt = _weight_tensor(node)
        if wt is not None and wt in tensor_to_entry:
            node_idx_to_entry[node_idx] = tensor_to_entry[wt]
            continue
        if pos >= len(all_names):
            raise RuntimeError(
                f"ran out of conv_order.json entries at node_idx={node_idx} (pos={pos}, "
                f"available={len(all_names)}) -- positional correspondence broken, do not proceed."
            )
        entry = all_names[pos]
        pos += 1
        if wt is not None:
            tensor_to_entry[wt] = entry
        node_idx_to_entry[node_idx] = entry
    if pos != len(all_names):
        raise RuntimeError(
            f"consumed only {pos}/{len(all_names)} conv_order.json entries -- positional "
            "correspondence broken, do not proceed."
        )

    result = {i: ([], []) for i in range(8)}
    for node_idx in weight_like_idx:
        pid = None
        for i in range(8):
            lo, hi = partition_node_index_range(i, boundaries)
            if lo <= node_idx and (hi is None or node_idx < hi):
                pid = i
                break
        assert pid is not None, f"node_idx {node_idx} not covered by any partition range"
        entry = node_idx_to_entry[node_idx]
        if "MaxPool" in entry["module_type"]:
            result[pid][1].append(entry["logical_name"])
        else:
            result[pid][0].append(entry["logical_name"])
    return result


def resolve_folding_entry(logical_name, per_layer):
    if logical_name in per_layer:
        return per_layer[logical_name], logical_name
    if logical_name.endswith(".conv.0"):
        stripped = logical_name[: -len(".0")]
        if stripped in per_layer:
            return per_layer[stripped], stripped
    return None, None


def derive_fallback_pe_simd(logical_name, per_layer, node=None):
    """For main_up/shortcut_proj nodes HAWQ's folding search never saw (they
    don't exist in the trainable model -- FINN-export-only additions, both
    frozen plain nn.Conv2d/nn.ConvTranspose2d, never appear as MVAU/VVAU
    nodes at all). Kept for parity with the separable-family bridge; in
    practice this should never fire for this architecture since every real
    MVAU/VVAU-producing module here has a per_layer entry (including the new
    up4/up5.skip_resize_conv.0 dense conv, which HAS its own per_layer
    entry -- unlike main_up, it's a real learned site)."""
    prefix = logical_name.split(".")[0]
    reduce_entry = per_layer.get(f"{prefix}.reduce.0")
    expand_entry = per_layer.get(f"{prefix}.expand.0")
    if reduce_entry is not None and expand_entry is not None:
        pe, simd = expand_entry["pe"], reduce_entry["simd"]
        source = f"{prefix}.{{reduce,expand}}.0"
        if node is not None:
            inst = getCustomOp(node)
            mh, mw = _get_pe_simd_bounds(inst)
            safe_pe, safe_simd = _largest_divisor_leq(mh, pe), _largest_divisor_leq(mw, simd)
            if (safe_pe, safe_simd) != (pe, simd):
                source += f" (clamped PE {pe}->{safe_pe} for MH={mh}, SIMD {simd}->{safe_simd} for MW={mw})"
            pe, simd = safe_pe, safe_simd
        return {"PE": pe, "SIMD": simd}, source
    return {"PE": 1, "SIMD": 1}, None


def _largest_divisor_leq(n, cap):
    cap = max(1, min(cap, n))
    for d in range(cap, 0, -1):
        if n % d == 0:
            return d
    return 1


def _smallest_divisor_geq(n, minimum):
    """Smallest divisor of n that is >= minimum -- used to scale a VVAU's PE
    UP to match its SWU's SIMD (the SWU's own SIMD nodeattr always already
    divides IFMChannels==n by construction, so this only has to search
    upward, never fails to find n itself in the worst case)."""
    minimum = max(1, min(minimum, n))
    for d in range(minimum, n + 1):
        if n % d == 0:
            return d
    return n


def _get_pe_simd_bounds(inst):
    try:
        return inst.get_nodeattr("MH"), inst.get_nodeattr("MW")
    except AttributeError:
        pass
    k_h, k_w = inst.get_nodeattr("Kernel")
    return inst.get_nodeattr("Channels"), k_h * k_w


def find_preceding_swu_fmpad(kernel_model, vvau_node):
    """Uses real tensor connectivity (not node-list position) -- kept for
    parity with the _v4 bridge this was copied from; should never actually
    be called for this architecture (0 VVAU nodes, see SWU_OP_TYPES comment
    above)."""
    swu_node = kernel_model.find_producer(vvau_node.input[0])
    if swu_node is None or swu_node.op_type not in SWU_OP_TYPES:
        raise RuntimeError(f"{vvau_node.name}: expected an SWU node feeding its main input, got "
                            f"{getattr(swu_node, 'op_type', None)} ({getattr(swu_node, 'name', None)})")
    fmpad_node = kernel_model.find_producer(swu_node.input[0])
    if fmpad_node is not None and fmpad_node.op_type not in FMPAD_OP_TYPES:
        fmpad_node = None  # this SWU needs no explicit padding (e.g. padding=0) -- not an error
    return fmpad_node, swu_node


def find_following_thresholding(kernel_model, weight_node):
    """standalone Thresholding_hls/_rtl node directly consuming this MVAU/
    VVAU's output (noActivation=1 forced by step_enet_convert_to_hw_rtl_mvau,
    so activation is never fused into the weight node here)."""
    consumer = kernel_model.find_consumer(weight_node.output[0])
    if consumer is not None and consumer.op_type in THRESH_OP_TYPES:
        return consumer
    return None


def build_partition_folding_config(preamble_dir, partition_idx, sdp_node_name, partition_model_fn, logical_names, pool_names, per_layer, output_dir):
    kernel_model = ModelWrapper(partition_model_fn)
    print(f"[partition {partition_idx}] loaded raw model: {len(kernel_model.graph.node)} nodes")

    dummy_cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=output_dir)
    kernel_model = step_specialize_layers(kernel_model, dummy_cfg)
    kernel_model = kernel_model.transform(GiveUniqueNodeNames(sdp_node_name + "_"))
    kernel_model = kernel_model.transform(GiveReadableTensorNames())
    kernel_model = step_target_fps_parallelization(kernel_model, dummy_cfg)
    kernel_model = kernel_model.transform(GiveUniqueNodeNames())

    all_nodes = list(kernel_model.graph.node)
    name_to_idx = {n.name: i for i, n in enumerate(all_nodes)}
    weight_nodes = [n for n in all_nodes if n.op_type in WEIGHT_OP_TYPES]
    print(f"[partition {partition_idx}] {len(weight_nodes)} weight nodes vs {len(logical_names)} logical names "
          f"(+{len(pool_names)} pool-type, skipped: {pool_names})")
    if len(weight_nodes) != len(logical_names):
        print(f"[partition {partition_idx}] MISMATCH -- FINN nodes: {[n.name + '/' + n.op_type for n in weight_nodes]}")
        print(f"[partition {partition_idx}] MISMATCH -- logical names: {logical_names}")
        raise RuntimeError(f"partition {partition_idx}: weight node count != logical name count, aborting.")

    folding_config = {"Defaults": {}}
    unmatched = []
    n_swu_fmpad = 0
    n_thresh = 0
    for node, logical_name in zip(weight_nodes, logical_names):
        entry, json_key = resolve_folding_entry(logical_name, per_layer)
        if entry is None:
            unmatched.append(logical_name)
            fallback, source = derive_fallback_pe_simd(logical_name, per_layer, node)
            folding_config[node.name] = fallback
            print(f"[partition {partition_idx}]  {node.name:30s} {node.op_type:12s} <- {logical_name:25s} "
                  f"(derived from {source}) PE={fallback['PE']} SIMD={fallback['SIMD']}")
            continue

        node_type = entry.get("node_type")  # None for a legacy solve_folding-shaped entry
        compute_entry = entry["vvau"] if node_type == "depthwise_vvau_slot" else entry
        pe, simd = compute_entry["pe"], compute_entry["simd"]
        inst = getCustomOp(node)
        mh, mw = _get_pe_simd_bounds(inst)

        # See SWU_OP_TYPES' own comment: this architecture has 0 depthwise
        # VVAU nodes, so this branch should never fire in practice -- kept
        # for parity/safety with the _v4 bridge code it was copied from.
        swu_node = fmpad_node = None
        if node.op_type in VVAU_OP_TYPES:
            fmpad_node, swu_node = find_preceding_swu_fmpad(kernel_model, node)
            swu_simd = getCustomOp(swu_node).get_nodeattr("SIMD")
            if swu_simd > pe:
                print(f"[partition {partition_idx}]  {node.name:30s} {node.op_type:12s} <- {logical_name:25s} "
                      f"({json_key:25s}) SCALING PE {pe}->{swu_simd} to match {swu_node.name}'s own SIMD "
                      f"(never shrinking the SWU down to the MILP's PE={pe})")
                pe = swu_simd

        safe_pe = _smallest_divisor_geq(mh, pe) if node.op_type in VVAU_OP_TYPES else _largest_divisor_leq(mh, pe)
        safe_simd = _largest_divisor_leq(mw, simd)
        if (safe_pe, safe_simd) != (pe, simd):
            print(f"[partition {partition_idx}]  {node.name:30s} {node.op_type:12s} <- {logical_name:25s} "
                  f"({json_key:25s}) CLAMPED PE {pe}->{safe_pe} (MH={mh}), SIMD {simd}->{safe_simd} (MW={mw})")
        pe, simd = safe_pe, safe_simd
        node_config = {"PE": pe, "SIMD": simd}
        # MVAU/VVAU WEIGHT memory ram_style is never forced here (left at
        # FINN's own "auto" default) -- per _v4's own convention, "ultra"
        # requires runtime_writeable_weights=1 (which this bridge never
        # sets) or CreateStitchedIP asserts. This is unrelated to the new
        # FIFO URAM allocation below (step_allocate_uram_fifos), which
        # targets StreamingFIFO_rtl nodes, not MVAU/VVAU weight memory, and
        # has no such requirement.
        ram_style = compute_entry.get("ram_style")
        if ram_style is not None and ram_style != "ultra":
            node_config["ram_style"] = ram_style
        if "mem_mode" in compute_entry:
            node_config["mem_mode"] = compute_entry["mem_mode"]
        folding_config[node.name] = node_config
        extra = "".join(f" {k}={v}" for k, v in node_config.items() if k not in ("PE", "SIMD"))
        print(f"[partition {partition_idx}]  {node.name:30s} {node.op_type:12s} <- {logical_name:25s} "
              f"({json_key:25s}) PE={pe} SIMD={simd}{extra}")

        # standalone Thresholding (noActivation=1 forced pre-partitioning)
        # gets its PE straight from this same entry's own "thr_pe" -- solved
        # jointly with pe/simd in the same ILP run, no fixup/clamping needed.
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
                print(f"[partition {partition_idx}]  {thresh_node.name:30s} {thresh_node.op_type:12s} <- {logical_name:25s} "
                      f"({json_key:25s}, thr_pe) PE={thr_pe}{thr_extra}")

        if swu_node is not None:
            swu_config = {"parallel_window": 1} if simd > 1 else {}
            if swu_config:
                folding_config[swu_node.name] = swu_config
            n_swu_fmpad += 1
            print(f"[partition {partition_idx}]  {swu_node.name:30s} {swu_node.op_type:12s} <- {logical_name:25s} "
                  f"(SWU feeding {node.name}, own SIMD={swu_simd}, VVAU PE={pe}) {swu_config or '(unchanged)'}")
    if unmatched:
        print(f"[partition {partition_idx}] WARNING: {len(unmatched)} unmatched logical names "
              f"(derived fallback PE/SIMD applied): {unmatched}")
    if n_swu_fmpad:
        print(f"[partition {partition_idx}] scaled {n_swu_fmpad} VVAU PE(s) up to their SWU's SIMD "
              f"(parallel_window forced where SIMD>1)")
    if n_thresh:
        print(f"[partition {partition_idx}] bridged {n_thresh} standalone Thresholding node(s) via pe_thr")

    # InferLabelSelectLayer always inserts with PE=1 (get_exp_cycles() =
    # Labels/PE = 5 cycles/pixel) -- 5x slower than every other node's ~1
    # cycle/pixel folding, making it the pipeline bottleneck if left alone.
    # Labels=5 is prime, so PE=5 (full parallelism, 1 cycle/pixel) is the
    # only alternative to PE=1 -- force it whenever this node is present
    # (only true once the argmax-in-PL export, with its trailing TopK, is
    # used as MODEL_NAME -- absent from today's dummy/no-argmax export).
    n_labelselect = 0
    for node in all_nodes:
        if node.op_type == "LabelSelect":
            labels = getCustomOp(node).get_nodeattr("Labels")
            pe = 5 if labels % 5 == 0 else 1
            folding_config[node.name] = {"PE": pe}
            n_labelselect += 1
            print(f"[partition {partition_idx}]  {node.name:30s} {node.op_type:12s} <- (argmax-in-PL) "
                  f"Labels={labels} PE={pe} "
                  f"({'1 cyc/px, fully parallel' if pe == labels else 'UNPARALLELIZED -- bottleneck!'})")
    if n_labelselect:
        print(f"[partition {partition_idx}] forced PE on {n_labelselect} LabelSelect node(s) (argmax-in-PL)")

    return folding_config, len(unmatched)


def step_force_dsp(model, cfg=None):
    """Force resType=dsp on every MVAU (pointwise/1x1) AND VVAU (depthwise
    KxK) node -- this architecture has 0 VVAU nodes at all (all-dense,
    non-separable dilated context, AND a dense not depthwise decoder), so in
    practice this only ever touches MVAU."""
    n_dsp = 0
    for node in model.graph.node:
        if "MVAU" in node.op_type or "VVAU" in node.op_type:
            getCustomOp(node).set_nodeattr("resType", "dsp")
            n_dsp += 1
    print(f"[step_force_dsp] forced resType=dsp on {n_dsp} MVAU/VVAU node(s), ram_style left at default (auto)")
    return model


def step_fix_weight_dtype_bipolar_bug(model, cfg=None):
    """FINN's own MinimizeWeightBitWidth.minimize_weight_bit_width() picks
    BIPOLAR based only on weights.min(), without verifying weights.max()
    also fits BIPOLAR's exact {-1, +1} set."""
    n_fixed = 0
    for node in model.graph.node:
        if node.op_type not in WEIGHT_OP_TYPES:
            continue
        inst = getCustomOp(node)
        wdt_name = inst.get_nodeattr("weightDataType")
        if wdt_name != "BIPOLAR":
            continue
        w = model.get_initializer(node.input[1])
        if w is None or np.all((w == -1.0) | (w == 1.0)):
            continue
        w_min, w_max = float(w.min()), float(w.max())
        for cand in ["INT2", "INT3", "INT4", "INT5", "INT6", "INT7", "INT8"]:
            dt = DataType[cand]
            if dt.allowed(w_min) and dt.allowed(w_max):
                inst.set_nodeattr("weightDataType", cand)
                print(f"[step_fix_weight_dtype_bipolar_bug] {node.name}: BIPOLAR -> {cand} "
                      f"(w_min={w_min}, w_max={w_max})")
                n_fixed += 1
                break
        else:
            raise RuntimeError(f"{node.name}: could not find a valid INT dtype for w_min={w_min}, w_max={w_max}")
    if n_fixed:
        print(f"[step_fix_weight_dtype_bipolar_bug] fixed {n_fixed} mis-assigned BIPOLAR weightDataType node(s)")
    return model


def _widen_standalone_mvau_acc_for_downstream_threshold(inst, node, model):
    """Mirror the `thresholds is not None` widening branch of FINN's own
    MatrixVectorActivation.minimize_accumulator_width(), but sourcing the
    threshold min/max from a SEPARATE downstream Thresholding_hls/_rtl node
    instead of a fused input[2] -- see step_minimize_bit_width_standalone_
    thresh_aware's docstring for the full root-cause rationale."""
    if node.op_type not in WEIGHT_OP_TYPES:
        return False
    if len(node.input) > 2 or not inst.get_nodeattr("noActivation"):
        return False  # fused-activation case, FINN's own logic already handles it correctly
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
    idt = inst.get_input_datatype()
    acc_min, acc_max = calculate_matvec_accumulator_range(weights, idt)
    min_thr, max_thr = float(thresholds.min()), float(thresholds.max())
    if min_thr >= acc_min and max_thr <= acc_max:
        return False  # real thresholds already fit the weight-derived range, nothing to fix
    orig_min, orig_max = acc_min, acc_max
    acc_min = min(acc_min, min_thr)
    acc_max = max(acc_max, max_thr)
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
    print(f"[standalone-threshold acc widen] {node.name}: weight/input-only accumulator range "
          f"[{orig_min}, {orig_max}] was too narrow/wrongly-signed for downstream "
          f"{thresh_node.name}'s real threshold range [{min_thr}, {max_thr}] "
          f"-- set accDataType=outputDataType={adt.name}")
    return True


def step_minimize_bit_width_standalone_thresh_aware(model, cfg):
    """Drop-in replacement for finn.builder.build_dataflow_steps.step_minimize_bit_width
    that additionally accounts for downstream STANDALONE Thresholding_hls/_rtl
    consumers when computing an MVAU/VVAU node's accumulator/output dtype."""
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
        print(f"[step_minimize_bit_width_standalone_thresh_aware] widened {n_widened} "
              f"standalone-threshold MVAU/VVAU node(s) to cover their downstream "
              f"Thresholding node's real threshold range")
    return model


def _uram_blocks_for_fifo(inst, depth):
    """URAM288 tiling: each block is 4096-deep x 72b -- ceil(width/72) *
    ceil(depth/4096) blocks needed to hold one FIFO of this width/depth."""
    width = inst.get_instream_width()
    return math.ceil(width / URAM_WIDTH_PER_BLOCK) * math.ceil(depth / URAM_DEPTH_PER_BLOCK)


def step_allocate_uram_fifos(model, partition_idx, budget=URAM_BUDGET_BLOCKS):
    """Greedily allocates URAM to the deepest StreamingFIFO_rtl nodes
    (impl_style=="vivado" only -- impl_style=="rtl"/Q_srl FIFOs are shallow
    <=256-deep SRL chains with no BRAM/URAM primitive at all) up to a hard
    budget of `budget` URAM288 blocks (4096x72b each on UltraScale+).

    MUST run before SplitLargeFIFOs: confirmed via direct FINN source read
    (set_fifo_depths.py's SplitLargeFIFOs.apply()) that a FIFO's ram_style
    nodeattr is copied VERBATIM onto every one of its split children -- so
    allocating here, pre-split, automatically keeps a whole deep FIFO
    "chain" together in one contiguous URAM allocation once it gets split,
    with zero extra chain/adjacency-detection logic needed. Deepest-first
    greedy selection (per explicit instruction, "allocate to the deepest
    FIFO chains, especially adjacent not fragmented") -- a later (shallower)
    candidate is still tried if an earlier, deeper one didn't fit the
    remaining budget."""
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
          f"to {len(allocated)}/{len(candidates)} vivado-impl FIFO(s):")
    for name, depth, blocks in allocated:
        print(f"  {name:30s} depth={depth:8d} blocks={blocks}")
    return model


def _build_and_synth_one_partition(dataflow_model_filename, cfg, prefix, folding_config_file, partition_idx, report_dir):
    # Own subfolder per partition (e.g.
    # finn_build_tmp/S12_dense_nn_upsample_conv_v1/GenericPartition_3/) --
    # ProcessPoolExecutor reuses workers across multiple partitions, so this
    # must be derived from the constant _BASE_BUILD_DIR, never os.environ.get(
    # "FINN_BUILD_DIR") (that would read back this same worker's PREVIOUS
    # partition's mutated value and nest one level deeper each call).
    part_build_dir = os.path.join(_BASE_BUILD_DIR, prefix.rstrip("_"))
    os.makedirs(part_build_dir, exist_ok=True)
    os.environ["FINN_BUILD_DIR"] = part_build_dir

    part_cfg = dataclasses.replace(cfg, folding_config_file=folding_config_file)

    kernel_model = ModelWrapper(dataflow_model_filename)
    kernel_model = step_specialize_layers(kernel_model, part_cfg)
    # Unprefixed here -- step_apply_folding_config/step_set_fifo_depths call
    # FINN's own bare GiveUniqueNodeNames() internally and would wipe a prefix
    # set this early anyway. Prefix is (re-)applied right before IP-gen below,
    # matching FINN's own ZynqBuild.apply() reference ordering -- this is the
    # existing prefix-collision-avoidance fix, confirmed still sufficient
    # (this pipeline never combines partitions into one Vivado project, only
    # ever per-partition OOC synth) and kept unchanged.
    kernel_model = kernel_model.transform(GiveUniqueNodeNames())
    kernel_model = kernel_model.transform(GiveReadableTensorNames())
    kernel_model = step_target_fps_parallelization(kernel_model, part_cfg)
    kernel_model = step_apply_folding_config(kernel_model, part_cfg)
    kernel_model = step_minimize_bit_width_standalone_thresh_aware(kernel_model, part_cfg)
    kernel_model = step_fix_weight_dtype_bipolar_bug(kernel_model, part_cfg)
    kernel_model = step_force_dsp(kernel_model, part_cfg)
    kernel_model = step_hw_codegen(kernel_model, part_cfg)
    kernel_model = step_hw_ipgen(kernel_model, part_cfg)
    kernel_model = step_set_fifo_depths(kernel_model, part_cfg)
    kernel_model = step_allocate_uram_fifos(kernel_model, partition_idx)
    # FIFO-autosize checkpoint -- saved pre-split (ram_style already set,
    # depths already auto-sized) so it can be inspected without waiting for
    # SplitLargeFIFOs/ipgen/OOC synth to finish.
    autosize_ckpt = os.path.join(os.path.dirname(dataflow_model_filename), f"partition_{partition_idx}_prefifo_autosize.onnx")
    kernel_model.save(autosize_ckpt)
    print(f"[partition {partition_idx}] saved FIFO-autosize checkpoint: {autosize_ckpt}", flush=True)
    kernel_model = kernel_model.transform(SplitLargeFIFOs())
    # Applied here (not earlier) so it survives into the actual IP-XACT/Verilog
    # module names -- avoids identical module names across partitions when
    # combining the stitched IPs together in one Vivado project later.
    kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
    fpga_part = part_cfg._resolve_fpga_part()
    clk_period_ns = part_cfg.synth_clk_period_ns
    # SplitLargeFIFOs creates brand-new StreamingFIFO_rtl nodes with no
    # ip_path -- CreateStitchedIP asserts every node has one, so re-run ip
    # generation for them.
    kernel_model = kernel_model.transform(PrepareIP(fpga_part, clk_period_ns))
    kernel_model = kernel_model.transform(HLSSynthIP())
    kernel_model = kernel_model.transform(CreateStitchedIP(fpga_part, clk_period_ns, prefix.rstrip("_"), False))
    kernel_model.save(dataflow_model_filename)

    # Run this partition's OOC synth right here in this same worker process,
    # immediately after its own build finishes -- automatically advances to
    # OOC synth with no separate/deferred script needed; other partitions'
    # workers do the same concurrently, so no partition waits on the other 7.
    print(f"[partition {partition_idx}] build done, starting OOC synth...", flush=True)
    kernel_model = kernel_model.transform(SynthOutOfContext(part=fpga_part, clk_period_ns=clk_period_ns))
    res = eval(kernel_model.get_metadata_prop("res_total_ooc_synth"))
    kernel_model.save(dataflow_model_filename)
    report_path = os.path.join(report_dir, f"ooc_synth_partition_{partition_idx}.json")
    with open(report_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[partition {partition_idx}] OOC synth done: {res}", flush=True)
    return dataflow_model_filename, res


def step_build_all_partitions_with_folding_and_dsp(model, cfg, folding_config_map, parallel=True, max_workers=4):
    sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"

    report_dir = os.path.join(cfg.output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    jobs = []
    for i, sdp_node in enumerate(sdp_nodes):
        sdp_inst = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_inst.get_nodeattr("model")
        prefix = sdp_node.name + "_"
        jobs.append((dataflow_model_filename, prefix, folding_config_map[i], i))

    print("[step_build_all_partitions_with_folding_and_dsp] building+synthesizing %d partitions (parallel=%s)"
          % (len(jobs), parallel))

    ooc_results = {}
    if parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_build_and_synth_one_partition, fn, cfg, prefix, ffile, i, report_dir): i
                for fn, prefix, ffile, i in jobs
            }
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                _, res = fut.result()
                ooc_results[i] = res
                print("[step_build_all_partitions_with_folding_and_dsp] partition %d done" % i)
    else:
        for fn, prefix, ffile, i in jobs:
            _, res = _build_and_synth_one_partition(fn, cfg, prefix, ffile, i, report_dir)
            ooc_results[i] = res
            print("[step_build_all_partitions_with_folding_and_dsp] partition %d done" % i)

    all_results = {f"partition_{i}": ooc_results[i] for i in sorted(ooc_results)}
    numeric_keys = set()
    for res in all_results.values():
        for k, v in res.items():
            try:
                float(v)
                numeric_keys.add(k)
            except (TypeError, ValueError):
                pass
    aggregate = {}
    for k in numeric_keys:
        vals = [float(all_results[p][k]) for p in all_results if k in all_results[p]]
        if k.lower().startswith("fmax") or "period" in k.lower():
            aggregate[k + "_min_across_partitions"] = min(vals)
        else:
            aggregate[k + "_sum"] = sum(vals)
    all_results["aggregate"] = aggregate
    with open(os.path.join(report_dir, "ooc_synth_and_timing_per_partition.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("[step_build_all_partitions_with_folding_and_dsp] all partitions built+synthesized. Combined report:",
          os.path.join(report_dir, "ooc_synth_and_timing_per_partition.json"))

    return model


def main():
    # Persists across container recreation (bind-mounted host path), unlike
    # the container's own default FINN_BUILD_DIR=/tmp/finn_dev_thelegendiv.
    # Per-partition workers override this with their own GenericPartition_N
    # subfolder (see _build_and_synth_one_partition); this base value is only
    # used directly by main()'s own FINN calls (partitioning, combine step).
    os.environ["FINN_BUILD_DIR"] = _BASE_BUILD_DIR
    os.makedirs(_BASE_BUILD_DIR, exist_ok=True)

    if len(sys.argv) < 2:
        print("Usage: finn_ooc_12_dense_relu_nearest_conv_upsample_dummy_rtl_mvau_8way_full_v1_512x512.py "
              "<rtl_mvau_512x512_preamble_output_dir> [explicit_output_dir]")
        sys.exit(1)
    preamble_dir = sys.argv[1]
    flat_ckpt = os.path.join(preamble_dir, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
    print(f"Preamble dir: {preamble_dir}")
    print(f"Flat 8-way-tagged checkpoint: {flat_ckpt}")
    print(f"Conv order file: {CONV_ORDER_FILE}")
    print(f"Folding block file: {FOLDING_BLOCK_FILE}")

    with open(FOLDING_BLOCK_FILE) as f:
        per_layer = json.load(f)["per_layer"]

    if len(sys.argv) >= 3:
        OUTPUT_DIR = sys.argv[2]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR = os.path.join(
            base.ENET_DIR, "finn_deployment_outputs",
            f"12_dense_relu_nearest_conv_upsample_dummy_rtl_mvau_8way_full_v1_512x512_{timestamp}",
        )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"OUTPUT_DIR= {OUTPUT_DIR}", flush=True)

    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)
    flat_model = ModelWrapper(flat_ckpt)
    parent_model = step_create_dataflow_partition_multi(flat_model, cfg)
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    print(f"Got {len(sdp_nodes)} partitions: {[n.name for n in sdp_nodes]}")
    assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"
    validate_partition_single_output(parent_model)

    logical_by_partition = load_all_partition_logical_names(preamble_dir)
    folding_config_map = {}
    total_unmatched = 0
    for i, sdp_node in enumerate(sdp_nodes):
        conv_names, pool_names = logical_by_partition[i]
        partition_model_fn = getCustomOp(sdp_node).get_nodeattr("model")
        folding_config, n_unmatched = build_partition_folding_config(
            preamble_dir, i, sdp_node.name, partition_model_fn, conv_names, pool_names, per_layer, OUTPUT_DIR,
        )
        total_unmatched += n_unmatched
        out_path = os.path.join(OUTPUT_DIR, f"hawq_folding_config_partition{i}.json")
        with open(out_path, "w") as f:
            json.dump(folding_config, f, indent=2)
        folding_config_map[i] = out_path
        print(f"[partition {i}] saved bridged folding config ({len(folding_config) - 1} entries): {out_path}")

    print(f"\n=== Bridge summary: {total_unmatched} total unmatched logical names across all 8 partitions "
          "(expected: 0 for this architecture) ===\n")

    # Each partition is built AND OOC-synthesized inline inside its own
    # worker (see _build_and_synth_one_partition) -- no separate/deferred
    # per-partition synth script needed after this.
    parent_model = step_build_all_partitions_with_folding_and_dsp(
        parent_model, cfg, folding_config_map, parallel=True, max_workers=4,
    )
    parent_ckpt = os.path.join(OUTPUT_DIR, "intermediate_models", "dataflow_parent_built.onnx")
    os.makedirs(os.path.dirname(parent_ckpt), exist_ok=True)
    parent_model.save(parent_ckpt)

    cfg = dataclasses.replace(
        cfg,
        steps=[
            step_combine_partitions,
            step_generate_estimate_reports_multi,
            step_measure_rtlsim_performance_multi,
        ],
    )
    print("Proceeding to step_combine_partitions -> estimate reports -> rtlsim...", flush=True)
    build.build_dataflow_cfg(parent_ckpt, cfg)


if __name__ == "__main__":
    main()
