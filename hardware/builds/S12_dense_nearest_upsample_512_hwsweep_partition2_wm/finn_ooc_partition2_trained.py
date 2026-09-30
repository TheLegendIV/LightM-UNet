"""Partition-2-only real Vivado OOC synthesis for the
S12_dense_nearest_upsample_512_hwsweep_partition2_wm sweep -- adapted from
hardware/archive/finn_ooc_partition2_hawq_joint.py, parametrized by
<preamble_dir> and <tag>, with an OPTIONAL 3rd arg (bridged folding config
json). When the 3rd arg is omitted, step_apply_folding_config is skipped
entirely and FINN's own step_target_fps_parallelization auto-fold result is
left in place -- this is the "auto-fold" baseline_both_off variant.

No URAM: unlike the 8-way FULL builds (finn_ooc_..._8way_full_*.py), this
script has no step_allocate_uram_fifos call at all -- FIFOs stay at FINN's
default ram_style (BRAM/auto-inferred, never forced to "ultra"). This
matches the S19 partition-2-only template this is based on, which never had
that step either.

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_ooc_partition2_trained.py \\
        <hawq_preamble_output_dir> <tag> [<bridged_folding_config.json>]
"""

import math
import os
import sys
import dataclasses
from datetime import datetime

import numpy as np

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

if len(sys.argv) < 3:
    print("Usage: finn_ooc_partition2_trained.py <hawq_preamble_output_dir> <tag> [<bridged_folding_config.json>]")
    sys.exit(1)
HAWQ_PREAMBLE_DIR = sys.argv[1]
TAG = sys.argv[2]
BRIDGED_FOLDING_CONFIG = sys.argv[3] if len(sys.argv) > 3 else None

from qonnx.core.datatype import DataType  # noqa: E402
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402
from qonnx.transformation.infer_datatypes import InferDataTypes  # noqa: E402
from qonnx.util.basic import calculate_matvec_accumulator_range, roundup_to_integer_multiple  # noqa: E402

# finn_enet_ip_build_partitioned_8way reads sys.argv[1]/[2] at import time --
# mask our own CLI args during the import (see finn_hawq_folding_bridge.py's
# identical guard).
_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from finn_partition_build_steps import step_create_dataflow_partition_multi  # noqa: E402
from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
    step_measure_rtlsim_performance,
    step_out_of_context_synthesis,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP  # noqa: E402
from finn.transformation.fpgadataflow.set_fifo_depths import SplitLargeFIFOs  # noqa: E402
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth  # noqa: E402
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP  # noqa: E402
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP  # noqa: E402
from finn.util.fpgadataflow import is_fpgadataflow_node  # noqa: E402

PARTITION_IDX = 2
WEIGHT_OP_TYPES = ("MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl")
THRESH_OP_TYPES = ("Thresholding_hls", "Thresholding_rtl")


def step_force_dsp(model, cfg=None):
    """Force resType=dsp on every MVAU (pointwise/1x1) AND VVAU (depthwise
    KxK) node -- matches this sweep's MILP solve, which used --force-dsp
    (see MILP/artifacts/.../run_args.json), so the built hardware's resource
    type matches what the MILP already assumed when folding PE/SIMD."""
    n_dsp = 0
    for node in model.graph.node:
        if "MVAU" in node.op_type or "VVAU" in node.op_type:
            getCustomOp(node).set_nodeattr("resType", "dsp")
            n_dsp += 1
    print(f"[step_force_dsp] forced resType=dsp on {n_dsp} MVAU/VVAU node(s)")
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
    instead of a fused input[2] (this preamble forces noActivation=1 +
    standalone Thresholding on every MVAU/VVAU pre-partitioning)."""
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

SOURCE_CKPT = os.path.join(HAWQ_PREAMBLE_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
fold_suffix = "milpfold" if BRIDGED_FOLDING_CONFIG else "autofold"
OUTPUT_DIR = os.path.join(
    base.ENET_DIR, "finn_deployment_outputs",
    f"S12_dense_nearest_upsample_512_hwsweep_wm_{TAG}_{fold_suffix}_partition2_{timestamp}",
)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    assert os.path.exists(SOURCE_CKPT), f"missing {SOURCE_CKPT}"
    if BRIDGED_FOLDING_CONFIG:
        assert os.path.exists(BRIDGED_FOLDING_CONFIG), f"missing {BRIDGED_FOLDING_CONFIG}"

    # split_large_fifos is left at its default False here -- SplitLargeFIFOs()
    # is invoked manually below (matching finn_ooc_..._v4_...8way_full_*.py's
    # proven pattern) so a checkpoint can be saved BEFORE the split, right
    # after the expensive FIFO-autosizing rtlsim step. impl_style="vivado"
    # FIFOs cap at depth 32768 -- without splitting, CreateStitchedIP crashes
    # on any FIFO sized above that.
    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)

    print(f"Tag               : {TAG}")
    print(f"Fold mode         : {fold_suffix}")
    print(f"Source checkpoint : {SOURCE_CKPT}")
    print(f"Folding config    : {BRIDGED_FOLDING_CONFIG}")
    print(f"Output dir        : {OUTPUT_DIR}")

    flat_model = ModelWrapper(SOURCE_CKPT)

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

    if BRIDGED_FOLDING_CONFIG:
        cfg = dataclasses.replace(cfg, folding_config_file=BRIDGED_FOLDING_CONFIG)
        print("Running: step_apply_folding_config (MILP-bridged config)")
        kernel_model = step_apply_folding_config(kernel_model, cfg)
    else:
        print("Skipping step_apply_folding_config -- auto-fold variant, keeping "
              "step_target_fps_parallelization's own PE/SIMD choice.")

    print("Running: step_minimize_bit_width_standalone_thresh_aware")
    kernel_model = step_minimize_bit_width_standalone_thresh_aware(kernel_model, cfg)
    print("Running: step_fix_weight_dtype_bipolar_bug")
    kernel_model = step_fix_weight_dtype_bipolar_bug(kernel_model, cfg)
    print("Running: step_force_dsp")
    kernel_model = step_force_dsp(kernel_model, cfg)
    print("Running: step_hw_codegen")
    kernel_model = step_hw_codegen(kernel_model, cfg)
    print("Running: step_hw_ipgen")
    kernel_model = step_hw_ipgen(kernel_model, cfg)
    print("Running: step_set_fifo_depths")
    kernel_model = step_set_fifo_depths(kernel_model, cfg)

    # FIFO-autosize checkpoint -- saved pre-split so a future crash (in
    # SplitLargeFIFOs/PrepareIP/HLSSynthIP/CreateStitchedIP/OOC synth) can
    # resume from here without re-running the expensive rtlsim autosizing.
    autosize_ckpt = os.path.join(OUTPUT_DIR, f"partition2_{TAG}_{fold_suffix}_prefifo_autosize.onnx")
    kernel_model.save(autosize_ckpt)
    print(f"Saved FIFO-autosize checkpoint: {autosize_ckpt}")

    print("Running: SplitLargeFIFOs")
    kernel_model = kernel_model.transform(SplitLargeFIFOs())
    # Re-applied here (not just once, early) so it survives into the actual
    # Verilog module names for the brand-new split-FIFO nodes too.
    kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
    fpga_part = cfg._resolve_fpga_part()
    clk_period_ns = cfg.synth_clk_period_ns
    # SplitLargeFIFOs's new StreamingFIFO_rtl nodes have no ip_path yet --
    # CreateStitchedIP asserts every node has one, so (re-)generate IP for them.
    kernel_model = kernel_model.transform(PrepareIP(fpga_part, clk_period_ns))
    kernel_model = kernel_model.transform(HLSSynthIP())

    print("Running: CreateStitchedIP")
    # NOTE: deliberately NOT passing prefix as ip_name here (unlike v4's
    # per-partition build, which feeds custom-named per-partition stitched
    # IPs into step_combine_partitions + step_measure_rtlsim_performance_multi
    # afterward). This script is a standalone single-partition build that
    # calls the OFFICIAL finn.builder.build_dataflow_steps.step_measure_
    # rtlsim_performance directly, whose verilator_fifosim() has the
    # "finn_design_wrapper"/"finn_design_wrapper.v" module/file name HARDCODED
    # (see finn/src/finn/util/pyverilator.py). FINN's own step_create_stitched_ip
    # never overrides ip_name either -- it always uses the CreateStitchedIP
    # default ("finn_design"). Passing a custom ip_name here silently breaks
    # rtlsim: Verilator fails with "Cannot find file containing module:
    # finn_design_wrapper.v" because the real wrapper is named
    # "<prefix>_wrapper.v" instead.
    kernel_model = kernel_model.transform(
        CreateStitchedIP(fpga_part, clk_period_ns)
    )
    final_fn = os.path.join(OUTPUT_DIR, f"partition2_{TAG}_{fold_suffix}_stitched.onnx")
    kernel_model.save(final_fn)

    # OOC synth runs first and independently of rtlsim (both only need the
    # stitched IP, neither depends on the other's output) -- this guarantees
    # we get resource/timing numbers even if rtlsim hits an unrelated issue.
    print("Running: step_out_of_context_synthesis")
    kernel_model = step_out_of_context_synthesis(kernel_model, cfg)
    kernel_model.save(final_fn)

    print("Running: step_measure_rtlsim_performance")
    try:
        kernel_model = step_measure_rtlsim_performance(kernel_model, cfg)
        kernel_model.save(final_fn)
    except Exception as e:
        print(f"[step_measure_rtlsim_performance] FAILED, continuing without it: {e}")

    print(f"Done. Reports in {OUTPUT_DIR}/report")
