"""ONE consolidated task: for EVERY one of the 8 stage-based FINN partitions
of the REAL, QAT fine-tuned checkpoint_best.pth (epoch 50) weight transfer,
per-block-HAWQ-bit-width (min4 scheme), separable_dilated + dense_dilation,
plain-ReLU 12_separable_dense_relu export, derive its bridged (PE, SIMD,
resType=dsp, forced on both MVAU AND VVAU nodes -- this architecture happens
to have 0 VVAU/depthwise nodes, see compression/hawq/config_12_separable_
dense_relu.py's SEPARABLE_DILATED=True + USE_DSC=False, but step_force_dsp
is left generic) folding config directly from
compression/hawq/artifacts/folding_block_12_separable_dense_relu_min4_hardcap131_maxspeed.json
(per-layer ILP output at a RELAXED 131% LUT hard-cap vs the dummy-weight
build's 100% hardcap100_maxspeed file -- same per_layer logical-dotted-name
schema, resolve_folding_entry()/build_partition_folding_config() below are
schema-generic and need zero changes for this), then run the full
per-partition build (specialize -> fold -> apply HAWQ folding -> force DSP
-> HLS/RTL codegen -> ipgen -> FIFO depths -> stitched IP) for all 8
partitions, combine them, and generate estimate/rtlsim reports -- all as a
SINGLE build_dataflow_cfg() call/process.

Byte-for-byte copy of finn_ooc_12_separable_dense_relu_min4_8way_full.py
(the DUMMY-weight, hardcap100 sibling) with only MODEL_NAME/CONV_ORDER_FILE/
FOLDING_BLOCK_FILE/output-dir naming changed, PLUS one addition: main() now
accepts an OPTIONAL 2nd CLI arg (an explicit OUTPUT_DIR) so a wrapper shell
command can compute the timestamped dir ONCE and pass the SAME path to both
this script and the per-partition OOC-synth script for reliable `&&`
chaining (see module docstring's Run block below) -- avoids the previous
workflow's manual "wait for this script's own printed OUTPUT_DIR, then
hand-edit the synth script's hardcoded constant" step.

CONV_ORDER_FILE deliberately still points at the DUMMY model's own conv_order
sidecar (quantEnet_12_separable_dense_relu_min4_hawq_dummy_int8_conv_order.json,
produced by finn_hawq_dump_conv_order_12_separable_dense_relu.py) -- that
sidecar only encodes the ARCHITECTURE's named_modules() forward-order
(logical dotted names <-> positional weight-like-node index), which is
identical between the dummy and trained checkpoints (same topology, only
weight VALUES differ); no need to regenerate it.

Same DELIBERATE difference as the dummy-weight sibling: this file's own
steps list stops after step_measure_rtlsim_performance_multi and does NOT
attempt step_out_of_context_synthesis_multi at all (that combined-design
path has hit 3 distinct Vivado merge bugs across sessions -- see
/memories/repo/finn_gotchas.md). OOC synthesis is done exclusively by
finn_ooc_12_separable_dense_relu_min4_8way_per_partition_synth.py, invoked
automatically right after this script via the `&&` chain below (not a
separate manual step).

Resumes from the ALREADY-COMPLETED trained-weight preamble's
(finn_hawq_preamble_12_separable_dense_relu_min4_trained.py)
`assign_stage_partition_ids_8way.onnx` checkpoint.

Run inside the FINN container (after the trained preamble has completed),
chaining full-build -> per-partition OOC synth automatically via `&&` so
OOC synth reliably starts the moment the build finishes, unattended:
    docker exec -e HOME=/tmp/home_dir <container> bash -c \\
        'cd /home/thelegendiv/finn/notebooks/enet && \\
         OUTDIR=finn_deployment_outputs/hawq_12_sep_dense_relu_min4_trained_hardcap131_8way_full_$(date +%Y%m%d_%H%M%S) && \\
         nohup bash -c "python3 finn_ooc_12_separable_dense_relu_min4_trained_hardcap131_8way_full.py \\
           <preamble_output_dir> /home/thelegendiv/finn/notebooks/enet/\$OUTDIR && \\
           python3 finn_ooc_12_separable_dense_relu_min4_8way_per_partition_synth.py \\
           /home/thelegendiv/finn/notebooks/enet/\$OUTDIR" \\
         > /tmp/hawq_12_sep_dense_relu_min4_trained_hardcap131_8way_full_and_synth.log 2>&1 &'
"""
import concurrent.futures
import dataclasses
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402

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
from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
)
from finn_partition_build_steps import (  # noqa: E402
    step_create_dataflow_partition_multi,
    step_combine_partitions,
    step_generate_estimate_reports_multi,
    step_measure_rtlsim_performance_multi,
)

MODEL_NAME = "quantEnet_12_separable_dense_relu_min4_hawq_trained_int8"
CONV_ORDER_FILE = os.path.join(base.ENET_DIR, "quantEnet_12_separable_dense_relu_min4_hawq_dummy_int8_conv_order.json")
FOLDING_BLOCK_FILE = os.path.join(
    base.ENET_DIR, "folding_block_12_separable_dense_relu_min4_hardcap131_maxspeed.json"
)
WEIGHT_OP_TYPES = ("MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl")
VVAU_OP_TYPES = ("VVAU_hls", "VVAU_rtl")
# Real FINN node types for the sliding-window unit and its preceding padding
# block, feeding a VVAU -- see compression/hawq/folding_ilp.py's
# solve_folding_nodewise, whose "depthwise_vvau_slot" output entries this
# bridge also writes onto these two node types when present (this
# architecture has 0 depthwise layers, so in practice these never fire --
# kept for parity with the w16/w20-family bridge logic).
SWU_OP_TYPES = ("ConvolutionInputGenerator_hls", "ConvolutionInputGenerator_rtl")
FMPAD_OP_TYPES = ("FMPadding_hls", "FMPadding_rtl", "FMPadding_Pixel")

PARTITION_RANGE_ORDER = [
    "down1_start", "down2_start", "q2_start", "q3_start", "q4_start", "up4_start", "up5_start",
]


def partition_node_index_range(partition_idx, boundaries):
    edges = [0] + [boundaries[k] for k in PARTITION_RANGE_ORDER] + [None]
    return edges[partition_idx], edges[partition_idx + 1]


def load_all_partition_logical_names(preamble_dir):
    pre_partition_ckpt = os.path.join(preamble_dir, "intermediate_models", "step_enet_convert_to_hw.onnx")
    full_model = ModelWrapper(pre_partition_ckpt)

    boundaries = compute_8way_boundaries(full_model)
    print(f"[bridge] 8-way boundaries: {boundaries}")

    with open(CONV_ORDER_FILE) as f:
        all_names = json.load(f)

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

    result = {i: ([], []) for i in range(8)}
    for pos, node_idx in enumerate(weight_like_idx):
        pid = None
        for i in range(8):
            lo, hi = partition_node_index_range(i, boundaries)
            if lo <= node_idx and (hi is None or node_idx < hi):
                pid = i
                break
        assert pid is not None, f"node_idx {node_idx} not covered by any partition range"
        entry = all_names[pos]
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
    don't exist in the trainable model -- FINN-export-only additions).
    See finn_ooc_8_2_relu_no_reg_w16_acc2x_8way_full.py's identical helper
    for the full rationale (MW/MH clamp against dimensioning drift)."""
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


def _get_pe_simd_bounds(inst):
    try:
        return inst.get_nodeattr("MH"), inst.get_nodeattr("MW")
    except AttributeError:
        pass
    k_h, k_w = inst.get_nodeattr("Kernel")
    return inst.get_nodeattr("Channels"), k_h * k_w


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
        safe_pe, safe_simd = _largest_divisor_leq(mh, pe), _largest_divisor_leq(mw, simd)
        if (safe_pe, safe_simd) != (pe, simd):
            print(f"[partition {partition_idx}]  {node.name:30s} {node.op_type:12s} <- {logical_name:25s} "
                  f"({json_key:25s}) CLAMPED PE {pe}->{safe_pe} (MH={mh}), SIMD {simd}->{safe_simd} (MW={mw})")
        pe, simd = safe_pe, safe_simd
        node_config = {"PE": pe, "SIMD": simd}
        # "ultra" (URAM) requires runtime_writeable_weights=1 (FINN's own
        # CreateStitchedIP assertion in matrixvectoractivation.py's
        # code_generation_ipi) which this bridge never sets -- rather than
        # force that extra nodeattr on, just don't force URAM at all here;
        # drop straight to FINN's own "auto" ram_style default instead.
        ram_style = compute_entry.get("ram_style")
        if ram_style is not None and ram_style != "ultra":
            node_config["ram_style"] = ram_style
        if "mem_mode" in compute_entry:
            node_config["mem_mode"] = compute_entry["mem_mode"]
        folding_config[node.name] = node_config
        extra = "".join(f" {k}={v}" for k, v in node_config.items() if k not in ("PE", "SIMD"))
        print(f"[partition {partition_idx}]  {node.name:30s} {node.op_type:12s} <- {logical_name:25s} "
              f"({json_key:25s}) PE={pe} SIMD={simd}{extra}")

        if node_type == "depthwise_vvau_slot":
            if entry["swu"]["simd"] != compute_entry["pe"] or entry["fmpadding"]["simd"] != entry["swu"]["simd"]:
                print(f"[partition {partition_idx}] WARNING: {logical_name}'s own ILP entry violates the "
                      f"fmpadding.simd==swu.simd==vvau.pe invariant BEFORE clamping "
                      f"(fmpadding={entry['fmpadding']['simd']}, swu={entry['swu']['simd']}, vvau.pe="
                      f"{compute_entry['pe']}) -- investigate compression/hawq/folding_ilp.py's own solve; "
                      f"forcing to the clamped VVAU PE below regardless.")
            fmpad_node, swu_node = find_preceding_swu_fmpad(all_nodes, name_to_idx, node)
            folding_config[swu_node.name] = {"SIMD": pe}
            folding_config[fmpad_node.name] = {"SIMD": pe}
            n_swu_fmpad += 1
            print(f"[partition {partition_idx}]  {swu_node.name:30s} {swu_node.op_type:12s} <- {logical_name:25s} "
                  f"(SWU, coupled to {node.name}'s PE) SIMD={pe}")
            print(f"[partition {partition_idx}]  {fmpad_node.name:30s} {fmpad_node.op_type:12s} <- {logical_name:25s} "
                  f"(FMPadding, chained to {swu_node.name}'s SIMD) SIMD={pe}")
    if unmatched:
        print(f"[partition {partition_idx}] WARNING: {len(unmatched)} unmatched logical names "
              f"(derived fallback PE/SIMD applied): {unmatched}")
    if n_swu_fmpad:
        print(f"[partition {partition_idx}] bridged {n_swu_fmpad} FMPadding+SWU pair(s) for depthwise VVAU slots")
    return folding_config, len(unmatched)


def step_force_dsp(model, cfg=None):
    """Force resType=dsp on every MVAU (pointwise/1x1) AND VVAU (depthwise
    KxK) node -- this architecture has 0 VVAU nodes (all-dense
    separable_dilated), so in practice this only ever touches MVAU."""
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
    also fits BIPOLAR's exact {-1, +1} set -- see finn_ooc_8_2_relu_no_reg_
    w16_acc2x_8way_full.py's identical helper for the full rationale."""
    import numpy as np
    from qonnx.core.datatype import DataType

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


def _build_one_partition_with_folding_and_dsp(dataflow_model_filename, cfg, prefix, folding_config_file):
    part_cfg = dataclasses.replace(cfg, folding_config_file=folding_config_file)

    kernel_model = ModelWrapper(dataflow_model_filename)
    kernel_model = step_specialize_layers(kernel_model, part_cfg)
    kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
    kernel_model = kernel_model.transform(GiveReadableTensorNames())
    kernel_model = step_target_fps_parallelization(kernel_model, part_cfg)
    kernel_model = step_apply_folding_config(kernel_model, part_cfg)
    kernel_model = step_minimize_bit_width(kernel_model, part_cfg)
    kernel_model = step_fix_weight_dtype_bipolar_bug(kernel_model, part_cfg)
    kernel_model = step_force_dsp(kernel_model, part_cfg)
    kernel_model = step_hw_codegen(kernel_model, part_cfg)
    kernel_model = step_hw_ipgen(kernel_model, part_cfg)
    kernel_model = step_set_fifo_depths(kernel_model, part_cfg)
    kernel_model = kernel_model.transform(
        CreateStitchedIP(part_cfg._resolve_fpga_part(), part_cfg.synth_clk_period_ns, prefix.rstrip("_"), False)
    )
    kernel_model.save(dataflow_model_filename)
    return dataflow_model_filename


def step_build_all_partitions_with_folding_and_dsp(model, cfg, folding_config_map, parallel=True, max_workers=4):
    sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"

    jobs = []
    for i, sdp_node in enumerate(sdp_nodes):
        sdp_inst = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_inst.get_nodeattr("model")
        prefix = sdp_node.name + "_"
        jobs.append((dataflow_model_filename, prefix, folding_config_map[i]))

    print("[step_build_all_partitions_with_folding_and_dsp] building %d partitions (parallel=%s)"
          % (len(jobs), parallel))

    if parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_build_one_partition_with_folding_and_dsp, fn, cfg, prefix, ffile): prefix
                for fn, prefix, ffile in jobs
            }
            for fut in concurrent.futures.as_completed(futures):
                prefix = futures[fut]
                fut.result()
                print("[step_build_all_partitions_with_folding_and_dsp] partition %s done" % prefix)
    else:
        for fn, prefix, ffile in jobs:
            _build_one_partition_with_folding_and_dsp(fn, cfg, prefix, ffile)
            print("[step_build_all_partitions_with_folding_and_dsp] partition %s done" % prefix)

    return model


def main():
    if len(sys.argv) < 2:
        print("Usage: finn_ooc_12_separable_dense_relu_min4_trained_hardcap131_8way_full.py "
              "<hawq_preamble_output_dir> [explicit_output_dir]")
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
            f"hawq_12_sep_dense_relu_min4_trained_hardcap131_8way_full_{timestamp}",
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
          "(expected: exactly the main_up/shortcut_proj ones -- inspect the per-partition prints above) ===\n")

    def _step_build_all(model, cfg):
        return step_build_all_partitions_with_folding_and_dsp(model, cfg, folding_config_map, parallel=True, max_workers=4)
    _step_build_all.__name__ = "step_build_all_partitions_with_folding_and_dsp"

    # Deliberately STOPS at step_measure_rtlsim_performance_multi -- NO
    # step_out_of_context_synthesis_multi here (see module docstring): OOC
    # synthesis for this build is done exclusively by the dedicated
    # per-partition script, one Vivado run at a time (sequential), never via
    # this combined-design path. That script is chained via `&&` right
    # after this one in the Run block above, so it starts automatically.
    cfg = dataclasses.replace(
        cfg,
        steps=[
            step_combine_partitions,
            step_generate_estimate_reports_multi,
            step_measure_rtlsim_performance_multi,
        ],
    )
    parent_model = step_build_all_partitions_with_folding_and_dsp(
        parent_model, cfg, folding_config_map, parallel=True, max_workers=4,
    )
    parent_ckpt = os.path.join(OUTPUT_DIR, "intermediate_models", "dataflow_parent_built.onnx")
    os.makedirs(os.path.dirname(parent_ckpt), exist_ok=True)
    parent_model.save(parent_ckpt)

    print("Proceeding to step_combine_partitions -> estimate reports -> rtlsim...", flush=True)
    build.build_dataflow_cfg(parent_ckpt, cfg)
    print("Done. Reports in:", os.path.join(OUTPUT_DIR, "report"))
    print("OUTPUT_DIR=", OUTPUT_DIR)
    print("Next: finn_ooc_12_separable_dense_relu_min4_8way_per_partition_synth.py will run "
          "automatically via the `&&` chain (or run it manually with this OUTPUT_DIR as its "
          "1st CLI arg if it wasn't chained).")


if __name__ == "__main__":
    main()
