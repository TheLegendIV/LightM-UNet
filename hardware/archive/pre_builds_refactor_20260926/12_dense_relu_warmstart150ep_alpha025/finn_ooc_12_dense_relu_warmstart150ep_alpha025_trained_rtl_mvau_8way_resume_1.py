"""Partitions 0,2-7 of finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_
mvau_8way_full.py's 2026-09-17 01:11:15 run all completed successfully (specialize->
fold->minimize_bit_width->codegen->ipgen->fifo->stitch, confirmed via on-disk
partition_N.onnx timestamps/sizes all advancing past the run's start time).
Partition 1 crashed after ~7h inside step_minimize_bit_width: AssertionError in
Thresholding.get_hw_compatible_threshold_tensor (`MVAU_rtl_1` concluded an unsigned
UINT15 accumulator/output dtype purely from weights+input dtype -- FINN's own
MinimizeAccumulatorWidth has no visibility into a SEPARATE downstream Thresholding
node's real threshold range, only a FUSED input[2] -- but Thresholding_rtl_0's real
calibrated thresholds span [-29423, 29193], genuinely requiring signed). Root cause
+ fix (step_minimize_bit_width_standalone_thresh_aware /
_widen_standalone_mvau_acc_for_downstream_threshold) now live in the main build
script (redeployed before this script runs).

This script rebuilds ONLY partition 1 (still in pristine/raw state on disk, never
saved) using the EXISTING output directory / folding config, then proceeds with
step_combine_partitions -> estimate reports -> rtlsim using all 8 (now fully built)
partitions -- same deliberate stop-before-OOC-synth as the main script (OOC synth
handled separately by finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_
partition_synth.py, run manually after this completes).
"""
import dataclasses
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

import finn.builder.build_dataflow as build  # noqa: E402
from finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full import (  # noqa: E402
    _build_one_partition_with_folding_and_dsp,
)
from finn_partition_build_steps import (  # noqa: E402
    step_combine_partitions,
    step_generate_estimate_reports_multi,
    step_measure_rtlsim_performance_multi,
)

OUTPUT_DIR = (
    "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
    "12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_20260917_011115"
)
PARENT_CKPT = os.path.join(OUTPUT_DIR, "intermediate_models", "dataflow_parent.onnx")
BUILT_PARENT_CKPT = os.path.join(OUTPUT_DIR, "intermediate_models", "dataflow_parent_built.onnx")


def main():
    print(f"OUTPUT_DIR= {OUTPUT_DIR}", flush=True)
    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)

    parent_model = ModelWrapper(PARENT_CKPT)
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"

    for i in [1]:
        sdp_node = sdp_nodes[i]
        sdp_inst = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_inst.get_nodeattr("model")
        prefix = sdp_node.name + "_"
        folding_config_file = os.path.join(OUTPUT_DIR, f"hawq_folding_config_partition{i}.json")
        print(f"[resume] rebuilding partition {i} ({sdp_node.name}) from {dataflow_model_filename}", flush=True)
        _build_one_partition_with_folding_and_dsp(dataflow_model_filename, cfg, prefix, folding_config_file)
        print(f"[resume] partition {i} done", flush=True)

    cfg = dataclasses.replace(
        cfg,
        steps=[
            step_combine_partitions,
            step_generate_estimate_reports_multi,
            step_measure_rtlsim_performance_multi,
        ],
    )
    parent_model.save(BUILT_PARENT_CKPT)
    print("Proceeding to step_combine_partitions -> estimate reports -> rtlsim...", flush=True)
    build.build_dataflow_cfg(BUILT_PARENT_CKPT, cfg)
    print("Done. Reports in:", os.path.join(OUTPUT_DIR, "report"))
    print("OUTPUT_DIR=", OUTPUT_DIR)
    print("Next: run finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_partition_synth.py "
          "manually with this OUTPUT_DIR as its 1st CLI arg.")


if __name__ == "__main__":
    main()
