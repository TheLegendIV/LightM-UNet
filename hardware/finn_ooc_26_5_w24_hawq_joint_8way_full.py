"""ONE consolidated task: for EVERY one of the 8 stage-based FINN partitions
of the fresh-weight, per-block-HAWQ-bit-width (JOINT scheme), per-block-
leaky-ReLU 26_5_w24 export, derive its bridged (PE, SIMD, resType=dsp)
folding config directly from
compression/hawq/folding_block_26_5_w24_acc1x_joint_100pct.json, then run
the full per-partition build (specialize -> fold -> apply HAWQ folding ->
force DSP -> HLS/RTL codegen -> ipgen -> FIFO depths -> stitched IP) for
all 8 partitions, combine them, and run OOC synthesis -- all as a SINGLE
build_dataflow_cfg() call/process, so this is one nohup'able background
job instead of 8+ separate manual steps.

Resumes from the ALREADY-COMPLETED preamble's
`assign_stage_partition_ids_8way.onnx` checkpoint (tidy/streamline/
convert_to_hw/8-way partition-id tagging already done there, ~1hr wall
time -- not repeated here) instead of starting from the raw ONNX export.

Per-partition folding-bridge logic (node-index-range derivation, weight-
node/logical-name correspondence, ".conv.0"->".conv" key normalization,
"no match" skip-with-warning for shortcut_proj/*.pool logical names) is
copied from finn_hawq_folding_bridge.py's single-partition (PARTITION_IDX=2,
S19-only) version, generalized to loop partition_idx 0..7 and to use THIS
model's own conv_order.json + folding_block_*_100pct.json per_layer schema
(same key/naming convention, confirmed by inspection: "initial.conv",
"down1.pool", "stage2.0.conv.0"/"stage2.0.conv.3", etc.)

DSP-forcing (per user decision: leave ram_style at FINN's default "auto"
everywhere -- do NOT try to bridge/honor the ILP's ram_style/wm_uram18
intent -- but force resType="dsp" on every MVAU node to relieve LUT
pressure, given the 99.71%-of-budget razor-thin margin reported by the
100pct folding config's own diagnostics) is inserted as `step_force_dsp`
between step_minimize_bit_width and step_hw_codegen in each partition's
own build, mirroring finn_ooc_partition2_force_dsp_uram.py's
step_force_dsp_and_uram (minus the ram_style="ultra" part).

IMPORTANT SAFETY CHECK before this is allowed to proceed into hw_ipgen/
Vivado for all 8 partitions (~hours each): after generating each
partition's bridged folding config, this script prints the per-partition
match/mismatch summary (same as finn_hawq_folding_bridge.py always did) --
inspect that any "unmatched" set is limited to the KNOWN-expected ones
(shortcut_proj.0, *.pool) before trusting a long unattended run. If a
partition's weight-node count doesn't match its logical-name count, this
script aborts immediately (raises) rather than silently proceeding with a
partially-empty folding config.

Run inside the FINN container (after the preamble has completed):
    docker exec -e HOME=/tmp/home_dir <container> bash -c \
        "cd /home/thelegendiv/finn/notebooks/enet && nohup python3 \
        finn_ooc_26_5_w24_hawq_joint_8way_full.py <preamble_output_dir> \
        > /tmp/hawq_26_5_w24_joint_8way_full.log 2>&1 &"
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
    find_stage_boundaries,
    find_stage23_quarter_boundaries,
)

# finn_enet_ip_build_partitioned_8way reads sys.argv[1]/[2] (MODEL_NAME/
# FIFO_STRATEGY) at import time -- mask our own CLI args during the import
# so they aren't misread as that module's positional args (same fix as
# finn_hawq_folding_bridge.py needed).
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
    step_out_of_context_synthesis_multi,
)

MODEL_NAME = "quantEnet_26_5_w24_hawq_joint_int8"
CONV_ORDER_FILE = os.path.join(base.ENET_DIR, f"{MODEL_NAME}_conv_order.json")
FOLDING_BLOCK_FILE = os.path.join(base.ENET_DIR, "folding_block_26_5_w24_acc1x_joint_100pct.json")
WEIGHT_OP_TYPES = ("MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl")

# node-index range -> partition_id, mirrors assign_stage_partition_ids_8way.
PARTITION_RANGE_ORDER = [
    "down1_start", "down2_start", "q2_start", "q3_start", "q4_start", "up4_start", "up5_start",
]


def partition_node_index_range(partition_idx, boundaries):
    edges = [0] + [boundaries[k] for k in PARTITION_RANGE_ORDER] + [None]
    return edges[partition_idx], edges[partition_idx + 1]


def load_all_partition_logical_names(preamble_dir):
    """Returns {partition_idx: (conv_names, pool_names)} for all 8
    partitions in one pass over the pre-partition full graph + conv_order
    sidecar (same positional-correspondence validation as the original
    single-partition bridge, just done for every partition_idx at once)."""

    pre_partition_ckpt = os.path.join(preamble_dir, "intermediate_models", "step_enet_convert_to_hw.onnx")
    full_model = ModelWrapper(pre_partition_ckpt)

    down1_start, down2_start, up4_start, up5_start = find_stage_boundaries(full_model)
    q2_start, q3_start, q4_start = find_stage23_quarter_boundaries(down2_start, up4_start)
    boundaries = {
        "down1_start": down1_start, "down2_start": down2_start,
        "q2_start": q2_start, "q3_start": q3_start, "q4_start": q4_start,
        "up4_start": up4_start, "up5_start": up5_start,
    }
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


def derive_fallback_pe_simd(logical_name, per_layer):
    """For shortcut_proj/main_up nodes HAWQ's folding search never saw (they
    don't exist in the trainable model -- FINN-export-only additions):
    shortcut_proj/main_up share the exact same MW as their block's reduce.0
    (same block-input channel count) and the exact same MH as expand.0 (same
    block-output channel count), so borrowing SIMD from reduce.0 and PE from
    expand.0 is always divisibility-safe (both are already-valid divisors of
    that same MW/MH) and far better tuned than a blind PE=SIMD=1 fallback."""
    prefix = logical_name.split(".")[0]
    reduce_entry = per_layer.get(f"{prefix}.reduce.0")
    expand_entry = per_layer.get(f"{prefix}.expand.0")
    if reduce_entry is not None and expand_entry is not None:
        return {"PE": expand_entry["pe"], "SIMD": reduce_entry["simd"]}, f"{prefix}.{{reduce,expand}}.0"
    return {"PE": 1, "SIMD": 1}, None


def build_partition_folding_config(preamble_dir, partition_idx, sdp_node_name, partition_model_fn, logical_names, pool_names, per_layer, output_dir):
    """Re-derives partition_idx's kernel model (loaded from its own saved
    dataflow-partition .onnx path, taken directly from the SDP node's own
    'model' nodeattr -- not a guessed filename), runs specialize_layers +
    target_fps_parallelization + the UNPREFIXED GiveUniqueNodeNames()
    (matching what step_apply_folding_config will do internally), then
    zips its MVAU/VVAU-op-type nodes against `logical_names` to build the
    bridged {node_name: {"PE":.., "SIMD":..}} config. Returns
    (folding_config_dict, n_unmatched)."""

    kernel_model = ModelWrapper(partition_model_fn)
    print(f"[partition {partition_idx}] loaded raw model: {len(kernel_model.graph.node)} nodes")

    # output_dir must be a real, already-created dir -- step_target_fps_parallelization
    # writes auto_folding_config.json into it.
    dummy_cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=output_dir)
    kernel_model = step_specialize_layers(kernel_model, dummy_cfg)
    kernel_model = kernel_model.transform(GiveUniqueNodeNames(sdp_node_name + "_"))
    kernel_model = kernel_model.transform(GiveReadableTensorNames())
    kernel_model = step_target_fps_parallelization(kernel_model, dummy_cfg)
    kernel_model = kernel_model.transform(GiveUniqueNodeNames())  # unprefixed, matches step_apply_folding_config

    weight_nodes = [n for n in kernel_model.graph.node if n.op_type in WEIGHT_OP_TYPES]
    print(f"[partition {partition_idx}] {len(weight_nodes)} weight nodes vs {len(logical_names)} logical names "
          f"(+{len(pool_names)} pool-type, skipped: {pool_names})")
    if len(weight_nodes) != len(logical_names):
        print(f"[partition {partition_idx}] MISMATCH -- FINN nodes: {[n.name + '/' + n.op_type for n in weight_nodes]}")
        print(f"[partition {partition_idx}] MISMATCH -- logical names: {logical_names}")
        raise RuntimeError(f"partition {partition_idx}: weight node count != logical name count, aborting.")

    folding_config = {"Defaults": {}}
    unmatched = []
    for node, logical_name in zip(weight_nodes, logical_names):
        entry, json_key = resolve_folding_entry(logical_name, per_layer)
        if entry is None:
            unmatched.append(logical_name)
            # No HAWQ folding entry (e.g. main_up/shortcut_proj) -- derive a
            # divisibility-safe PE/SIMD from the same block's reduce.0/expand.0
            # entries instead of trusting FINN's auto target-fps folding (which
            # has been observed to pick an invalid SIMD for these nodes).
            fallback, source = derive_fallback_pe_simd(logical_name, per_layer)
            folding_config[node.name] = fallback
            print(f"[partition {partition_idx}]  {node.name:30s} {node.op_type:12s} <- {logical_name:25s} "
                  f"(derived from {source}) PE={fallback['PE']} SIMD={fallback['SIMD']}")
            continue
        folding_config[node.name] = {"PE": entry["pe"], "SIMD": entry["simd"]}
        print(f"[partition {partition_idx}]  {node.name:30s} {node.op_type:12s} <- {logical_name:25s} "
              f"({json_key:25s}) PE={entry['pe']} SIMD={entry['simd']}")
    if unmatched:
        print(f"[partition {partition_idx}] WARNING: {len(unmatched)} unmatched logical names "
              f"(derived fallback PE/SIMD applied): {unmatched}")
    return folding_config, len(unmatched)


def step_force_dsp(model, cfg=None):
    n_dsp = 0
    for node in model.graph.node:
        if "MVAU" in node.op_type:
            getCustomOp(node).set_nodeattr("resType", "dsp")
            n_dsp += 1
    print(f"[step_force_dsp] forced resType=dsp on {n_dsp} MVAU node(s), ram_style left at default (auto)")
    return model


def step_fix_weight_dtype_bipolar_bug(model, cfg=None):
    """FINN's own MinimizeWeightBitWidth.minimize_weight_bit_width() picks
    BIPOLAR based only on weights.min() (via DataType.get_smallest_possible),
    without verifying weights.max() also fits BIPOLAR's exact {-1, +1} set.
    Some HAWQ-quantized layers legitimately produce a {-1, 0} weight tensor
    (a valid 2-bit-range encoding, not true bipolar) -- this crashes later in
    step_hw_codegen's make_weight_file (array2hexstring assertion) with no
    node identity in the traceback. Detect any such mis-assigned BIPOLAR
    weightDataType and correct it to the smallest INT type that actually
    fits every observed weight value."""
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
        print("Usage: finn_ooc_26_5_w24_hawq_joint_8way_full.py <hawq_preamble_output_dir>")
        sys.exit(1)
    preamble_dir = sys.argv[1]
    flat_ckpt = os.path.join(preamble_dir, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
    print(f"Preamble dir: {preamble_dir}")
    print(f"Flat 8-way-tagged checkpoint: {flat_ckpt}")
    print(f"Conv order file: {CONV_ORDER_FILE}")
    print(f"Folding block file: {FOLDING_BLOCK_FILE}")

    with open(FOLDING_BLOCK_FILE) as f:
        per_layer = json.load(f)["per_layer"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(base.ENET_DIR, "finn_deployment_outputs", f"hawq_26_5_w24_joint_8way_full_{timestamp}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"OUTPUT_DIR= {OUTPUT_DIR}", flush=True)

    # ---- Step A: split into 8 partitions (reuses the preamble's own flat
    # checkpoint; must use the SAME cfg the real build will use so the
    # partition split is deterministic/identical). ----
    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)
    flat_model = ModelWrapper(flat_ckpt)
    parent_model = step_create_dataflow_partition_multi(flat_model, cfg)
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    print(f"Got {len(sdp_nodes)} partitions: {[n.name for n in sdp_nodes]}")
    assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"

    # ---- Step B: derive + save the bridged folding config for all 8
    # partitions BEFORE launching any real per-partition build. ----
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
          "(expected: exactly the shortcut_proj.0 / *.pool ones -- inspect the per-partition prints above) ===\n")

    # ---- Step C: full per-partition build (folding + DSP-forcing wired
    # in), then combine + estimate + rtlsim + real OOC synthesis. ----
    def _step_build_all(model, cfg):
        return step_build_all_partitions_with_folding_and_dsp(model, cfg, folding_config_map, parallel=True, max_workers=4)
    _step_build_all.__name__ = "step_build_all_partitions_with_folding_and_dsp"

    cfg = dataclasses.replace(
        cfg,
        steps=[
            step_combine_partitions,
            step_generate_estimate_reports_multi,
            step_measure_rtlsim_performance_multi,
            step_out_of_context_synthesis_multi,
        ],
    )
    # step_create_dataflow_partition_multi already ran above (Step A) to
    # derive logical-name correspondence; build_dataflow_cfg would just
    # redo an equivalent split if we let it, so instead we directly invoke
    # the per-partition build on `parent_model` ourselves, then hand the
    # already-partitioned model into build_dataflow_cfg's remaining steps
    # via its own step-runner (build.build_dataflow_cfg always takes a
    # model FILE, not an in-memory model) -- save parent_model and use it
    # as the entry point.
    parent_model = step_build_all_partitions_with_folding_and_dsp(
        parent_model, cfg, folding_config_map, parallel=True, max_workers=4,
    )
    parent_ckpt = os.path.join(OUTPUT_DIR, "intermediate_models", "dataflow_parent_built.onnx")
    os.makedirs(os.path.dirname(parent_ckpt), exist_ok=True)
    parent_model.save(parent_ckpt)

    print("Proceeding to step_combine_partitions -> estimate reports -> rtlsim -> OOC synthesis...", flush=True)
    build.build_dataflow_cfg(parent_ckpt, cfg)
    print("Done. Reports in:", os.path.join(OUTPUT_DIR, "report"))
    print("OUTPUT_DIR=", OUTPUT_DIR)


if __name__ == "__main__":
    main()
