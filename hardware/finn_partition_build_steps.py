"""Custom multi-partition build-step replacements for the 5-way stage split
(see finn_stage_partition.py for the partition_id assignment / boundary
detection logic).

These steps REPLACE, in the step list, the following single-partition
generic steps from finn.builder.build_dataflow_steps:
    step_create_dataflow_partition
    step_specialize_layers
    step_target_fps_parallelization
    step_apply_folding_config
    step_minimize_bit_width
    step_hw_codegen
    step_hw_ipgen
    step_set_fifo_depths
    step_create_stitched_ip

...with:
    step_create_dataflow_partition_multi   (this file)
    step_build_all_partitions              (this file; loops the *existing*
                                             generic per-partition steps
                                             above, unmodified, once per
                                             StreamingDataflowPartition,
                                             optionally in parallel)
    step_combine_partitions                (this file; new top-level
                                             stitch: 5 pre-packaged partition
                                             IPs + direct AXI-stream links,
                                             adapted from the interior-node
                                             wiring logic in
                                             MakeZYNQProject.apply(), minus
                                             its Zynq-PS/IODMA-specific
                                             branch which doesn't apply to
                                             our plain OOC-probe use case)

step_generate_estimate_reports/step_measure_rtlsim_performance/
step_out_of_context_synthesis are kept as-is in the caller's step list,
AFTER step_combine_partitions -- they operate on the model returned by
step_combine_partitions exactly as they did on the old single-stitch model.

NOTE: this is new, not-yet-run infrastructure code. It is grounded directly
in FINN's own proven multi-partition mechanisms (CreateDataflowPartition's
partition_id grouping, and MakeZYNQProject's interior-node direct-stream
wiring), but has not yet been executed against the real S19 model -- expect
to need at least one debug iteration once actually run.
"""

import os
import concurrent.futures

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames

from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.util.basic import make_build_dir

from finn.builder.build_dataflow_steps import (
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
)


def step_create_dataflow_partition_multi(model, cfg):
    """Same as build_dataflow_steps.step_create_dataflow_partition, minus
    the `assert len(sdp_nodes) == 1` restriction -- accepts any number of
    partitions (assumes assign_stage_partition_ids already ran and set
    partition_id on every node)."""

    parent_model = model.transform(
        CreateDataflowPartition(
            partition_model_dir=cfg.output_dir + "/intermediate_models/supported_op_partitions"
        )
    )
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    print("[step_create_dataflow_partition_multi] created %d partitions" % len(sdp_nodes))
    if cfg.save_intermediate_models:
        parent_model.save(cfg.output_dir + "/intermediate_models/dataflow_parent.onnx")
    return parent_model


def _build_one_partition(dataflow_model_filename, cfg, prefix):
    """Runs one partition's sub-model through the existing, unmodified
    generic per-node build steps, then packages it with CreateStitchedIP.
    Designed to be safe to run in a separate process (only takes/returns
    picklable filenames + cfg; cfg is a plain dataclass-like config
    object already proven to survive the same treatment in FINN's own
    parallel HLS-synth worker pool, NUM_DEFAULT_WORKERS)."""

    kernel_model = ModelWrapper(dataflow_model_filename)
    kernel_model = step_specialize_layers(kernel_model, cfg)
    kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
    kernel_model = kernel_model.transform(GiveReadableTensorNames())
    kernel_model = step_target_fps_parallelization(kernel_model, cfg)
    kernel_model = step_apply_folding_config(kernel_model, cfg)
    kernel_model = step_minimize_bit_width(kernel_model, cfg)
    kernel_model = step_hw_codegen(kernel_model, cfg)
    kernel_model = step_hw_ipgen(kernel_model, cfg)
    # this runs InsertAndSetFIFODepths, which internally does its own
    # throwaway measurement CreateStitchedIP + rtlsim pass, then applies
    # the measured depths -- see set_fifo_depths.py. Much cheaper now
    # since each partition is a fraction of the original 1665-cell size.
    kernel_model = step_set_fifo_depths(kernel_model, cfg)
    # final, real packaged stitched IP for this partition (small vlnv
    # name derived from prefix so each partition gets a distinct IP-XACT
    # component identity)
    kernel_model = kernel_model.transform(
        CreateStitchedIP(cfg._resolve_fpga_part(), cfg.synth_clk_period_ns, prefix.rstrip("_"), False)
    )
    kernel_model.save(dataflow_model_filename)
    return dataflow_model_filename


def step_build_all_partitions(model, cfg, parallel=True, max_workers=None):
    """Loops step_create_dataflow_partition_multi's resulting
    StreamingDataflowPartition nodes, building each one's sub-model
    independently through the full existing per-node pipeline
    (specialize -> fold -> codegen -> ipgen -> fifo-depths -> stitched IP).

    parallel=True dispatches partitions concurrently via
    ProcessPoolExecutor (each partition shells out to its own Vivado/HLS
    subprocess tree, so this is safe -- partitions share no state until
    step_combine_partitions runs afterward). Set parallel=False to debug
    one partition at a time if something goes wrong.
    """

    sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) > 0, "No StreamingDataflowPartition nodes found; did " \
        "assign_stage_partition_ids + step_create_dataflow_partition_multi run first?"

    jobs = []
    for sdp_node in sdp_nodes:
        sdp_inst = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_inst.get_nodeattr("model")
        prefix = sdp_node.name + "_"
        jobs.append((dataflow_model_filename, prefix))

    print("[step_build_all_partitions] building %d partitions (parallel=%s): %s"
          % (len(jobs), parallel, [p for _, p in jobs]))

    if parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_build_one_partition, fn, cfg, prefix): prefix
                for fn, prefix in jobs
            }
            for fut in concurrent.futures.as_completed(futures):
                prefix = futures[fut]
                fut.result()  # re-raises any exception from the worker
                print("[step_build_all_partitions] partition %s done" % prefix)
    else:
        for fn, prefix in jobs:
            _build_one_partition(fn, cfg, prefix)
            print("[step_build_all_partitions] partition %s done" % prefix)

    return model


def step_combine_partitions(model, cfg):
    """New top-level stitch: instantiates each partition's already-
    packaged stitched IP as a single Vivado BD cell, connects adjacent
    partitions directly via AXI-stream (no DMA/DDR -- this is a single
    on-chip design, not a multi-die/Alveo one), and exposes the first
    partition's unconnected input stream / last partition's unconnected
    output stream as top-level ports.

    Adapted from MakeZYNQProject.apply()'s interior-node wiring branch
    (direct connect_bd_intf_net between s_axis_i/m_axis_j), with the
    idma/odma/Zynq-PS/axi_interconnect/smartconnect machinery removed
    since we have no PS, no DMA, and no host AXI-lite config port to wire
    up for a plain OOC-probe build.
    """

    from finn.util.basic import get_vivado_root

    sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) > 0, "No partitions to combine"

    vivado_stitch_proj_dir = make_build_dir("combined_stitch_proj_")
    fpga_part = cfg._resolve_fpga_part()

    tcl = []
    tcl.append("create_project combined_stitch %s -part %s" % (vivado_stitch_proj_dir, fpga_part))
    instance_names = {}
    for node in model.graph.node:
        assert node.op_type == "StreamingDataflowPartition"
        sdp_inst = getCustomOp(node)
        dataflow_model_filename = sdp_inst.get_nodeattr("model")
        kernel_model = ModelWrapper(dataflow_model_filename)

        ipstitch_path = kernel_model.get_metadata_prop("vivado_stitch_proj")
        vlnv = kernel_model.get_metadata_prop("vivado_stitch_vlnv")
        assert ipstitch_path is not None and vlnv is not None, (
            "Partition %s has no stitched IP -- did step_build_all_partitions run?" % node.name
        )
        instance_names[node.name] = node.name
        tcl.append(
            "set_property ip_repo_paths [concat [get_property ip_repo_paths "
            "[current_project]] [list %s/ip]] [current_project]" % ipstitch_path
        )
        tcl.append("update_ip_catalog -rebuild -scan_changes")
        tcl.append("create_bd_cell -type ip -vlnv %s %s" % (vlnv, node.name))

    # direct stream connections between adjacent partitions (this network
    # is a strict linear chain, no branching between partitions)
    ordered = list(model.graph.node)
    for i in range(len(ordered) - 1):
        prod, cons = ordered[i], ordered[i + 1]
        tcl.append(
            "connect_bd_intf_net [get_bd_intf_pins %s/m_axis_0] "
            "[get_bd_intf_pins %s/s_axis_0]" % (instance_names[prod.name], instance_names[cons.name])
        )
    for name in instance_names.values():
        tcl.append("connect_bd_net [get_bd_pins %s/ap_clk] [get_bd_pins %s/ap_clk]" % (name, name))

    # expose first partition's input / last partition's output at top level
    first, last = instance_names[ordered[0].name], instance_names[ordered[-1].name]
    tcl.append("make_bd_intf_pins_external [get_bd_intf_pins %s/s_axis_0]" % first)
    tcl.append("make_bd_intf_pins_external [get_bd_intf_pins %s/m_axis_0]" % last)
    tcl.append("make_bd_pins_external [get_bd_pins %s/ap_clk]" % first)
    tcl.append("make_bd_pins_external [get_bd_pins %s/ap_rst_n]" % first)
    tcl.append("save_bd_design")
    tcl.append("validate_bd_design")

    tcl_path = os.path.join(vivado_stitch_proj_dir, "make_combined_project.tcl")
    with open(tcl_path, "w") as f:
        f.write("\n".join(tcl) + "\n")

    # NOTE: intentionally not auto-invoking vivado here yet -- run once
    # manually first (`vivado -mode batch -source make_combined_project.tcl`)
    # to validate the generated tcl before wiring this into an unattended
    # build, since this code path has not been exercised yet.
    model.set_metadata_prop("combined_stitch_proj", vivado_stitch_proj_dir)
    print("[step_combine_partitions] wrote %s -- run manually first to validate" % tcl_path)
    return model
