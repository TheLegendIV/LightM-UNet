"""Resume script: partitions 0-5 of the 26_9_w24 8-way build already
completed successfully (specialize->fold->codegen->ipgen->fifo->stitch, all
verified to have zero weight-dtype violations). Partitions 6 and 7 crashed
during step_hw_codegen with `AssertionError: This value is not permitted by
chosen dtype` -- root cause: FINN's own MinimizeWeightBitWidth mis-assigns
BIPOLAR to a {-1, 0} weight tensor (regular4.1.conv.0 / regular5.0.conv.0),
now fixed via step_fix_weight_dtype_bipolar_bug in the main build script.

This script rebuilds ONLY partitions 6 and 7 (still in pristine/raw state on
disk, never saved) using the EXISTING output directory / folding configs,
then proceeds with step_combine_partitions -> estimate reports -> rtlsim ->
OOC synthesis using all 8 (now fully built) partitions.
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
from finn_ooc_26_9_w24_ptq_joint_8way_full import (  # noqa: E402
    _build_one_partition_with_folding_and_dsp,
)
from finn_partition_build_steps import (  # noqa: E402
    step_combine_partitions,
    step_generate_estimate_reports_multi,
    step_measure_rtlsim_performance_multi,
    step_out_of_context_synthesis_multi,
)

OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/hawq_26_9_w24_ptq_8way_full_20260829_094221"
PARENT_CKPT = os.path.join(OUTPUT_DIR, "intermediate_models", "dataflow_parent.onnx")
BUILT_PARENT_CKPT = os.path.join(OUTPUT_DIR, "intermediate_models", "dataflow_parent_built.onnx")


def main():
    print(f"OUTPUT_DIR= {OUTPUT_DIR}", flush=True)
    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)

    parent_model = ModelWrapper(PARENT_CKPT)
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"

    for i in [6, 7]:
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
            step_out_of_context_synthesis_multi,
        ],
    )
    parent_model.save(BUILT_PARENT_CKPT)
    print("Proceeding to step_combine_partitions -> estimate reports -> rtlsim -> OOC synthesis...", flush=True)
    build.build_dataflow_cfg(BUILT_PARENT_CKPT, cfg)
    print("Done. Reports in:", os.path.join(OUTPUT_DIR, "report"))
    print("OUTPUT_DIR=", OUTPUT_DIR)


if __name__ == "__main__":
    main()
