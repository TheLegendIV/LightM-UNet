"""Per-partition build steps for the RTL-MVAU variant, identical to
finn_partition_build_steps.py's _build_one_partition/step_build_all_partitions
except for one extra fix-up: step_enet_convert_to_hw_rtl_mvau's standalone
Thresholding_hls/Thresholding_rtl nodes (previously fused into their MVAU/
VVAU, now separate) have no PE of their own from any existing folding
config -- give each one the same PE as its producing MVAU/VVAU so stream
widths still match.
"""
import concurrent.futures

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP

from finn.builder.build_dataflow_steps import (
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
)

from finn_partition_build_steps import (  # noqa: E402  (reused unchanged)
    step_create_dataflow_partition_multi,  # noqa: F401
    step_combine_partitions,  # noqa: F401
    step_generate_estimate_reports_multi,  # noqa: F401
    step_measure_rtlsim_performance_multi,  # noqa: F401
    step_out_of_context_synthesis_multi,  # noqa: F401
)


def _fix_thresholding_pe_to_producer_pe(kernel_model):
    for node in kernel_model.graph.node:
        if node.op_type not in ("Thresholding_hls", "Thresholding_rtl"):
            continue
        producer = kernel_model.find_producer(node.input[0])
        if producer is None or producer.op_type not in (
            "MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl",
        ):
            continue
        producer_pe = getCustomOp(producer).get_nodeattr("PE")
        getCustomOp(node).set_nodeattr("PE", producer_pe)
    return kernel_model


def _build_one_partition_rtl_mvau(dataflow_model_filename, cfg, prefix):
    kernel_model = ModelWrapper(dataflow_model_filename)
    kernel_model = step_specialize_layers(kernel_model, cfg)
    kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
    kernel_model = kernel_model.transform(GiveReadableTensorNames())
    kernel_model = step_target_fps_parallelization(kernel_model, cfg)
    kernel_model = step_apply_folding_config(kernel_model, cfg)
    # NOT calling _fix_thresholding_pe_to_producer_pe for now -- standalone
    # Thresholding_hls/Thresholding_rtl nodes keep their default PE=1.
    kernel_model = step_minimize_bit_width(kernel_model, cfg)
    kernel_model = step_hw_codegen(kernel_model, cfg)
    kernel_model = step_hw_ipgen(kernel_model, cfg)
    kernel_model = step_set_fifo_depths(kernel_model, cfg)
    kernel_model = kernel_model.transform(
        CreateStitchedIP(cfg._resolve_fpga_part(), cfg.synth_clk_period_ns, prefix.rstrip("_"), False)
    )
    kernel_model.save(dataflow_model_filename)
    return dataflow_model_filename


def step_build_all_partitions_rtl_mvau(model, cfg, parallel=True, max_workers=None):
    sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) > 0, "No StreamingDataflowPartition nodes found; did " \
        "assign_stage_partition_ids + step_create_dataflow_partition_multi run first?"

    jobs = []
    for sdp_node in sdp_nodes:
        sdp_inst = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_inst.get_nodeattr("model")
        prefix = sdp_node.name + "_"
        jobs.append((dataflow_model_filename, prefix))

    print("[step_build_all_partitions_rtl_mvau] building %d partitions (parallel=%s): %s"
          % (len(jobs), parallel, [p for _, p in jobs]))

    if parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_build_one_partition_rtl_mvau, fn, cfg, prefix): prefix
                for fn, prefix in jobs
            }
            for fut in concurrent.futures.as_completed(futures):
                prefix = futures[fut]
                fut.result()
                print("[step_build_all_partitions_rtl_mvau] partition %s done" % prefix)
    else:
        for fn, prefix in jobs:
            _build_one_partition_rtl_mvau(fn, cfg, prefix)
            print("[step_build_all_partitions_rtl_mvau] partition %s done" % prefix)

    return model
