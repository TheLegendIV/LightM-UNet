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
import re
import json
import concurrent.futures
import shutil
from copy import deepcopy
from functools import partial

import numpy as np

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.op_and_param_counts import (
    aggregate_dict_keys,
    op_and_param_counts,
)
from finn.analysis.fpgadataflow.res_estimation import (
    res_estimation,
    res_estimation_complete,
)
from finn.builder.build_dataflow_config import DataflowOutputType
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.util.basic import launch_process_helper, make_build_dir
from finn.util.pyverilator import prepare_stitched_ip_for_verilator

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


def _parse_wrapper_ports(wrapper_filename):
    """Parses a Vivado-generated *_wrapper.v file's port declarations
    (the simple `input/output [W-1:0]name;` style Vivado always emits for
    make_wrapper output), returning {port_name: width_str} where
    width_str is e.g. "[7:0]" or "" for a 1-bit signal."""

    with open(wrapper_filename) as f:
        text = f.read()
    port_widths = {}
    for m in re.finditer(r"^\s*(?:input|output)\s*(\[\d+:\d+\])?\s*(\w+)\s*;", text, re.MULTILINE):
        width, name = m.group(1), m.group(2)
        port_widths[name] = width or ""
    return port_widths


_SKIP_RENAME_SUBSTRINGS = ("regslice_core", "swg_pkg", "axis_infrastructure_v1_1_vl_rfs")
# regslice_core: finn.util.pyverilator.prepare_stitched_ip_for_verilator
# already special-cases this file (assumed genuinely content-identical
# across every instance) and keeps only one copy across the whole merged
# design by matching this substring in the file path -- renaming it would
# defeat that dedup. swg_pkg: same function unconditionally drops every
# swg_pkg.sv from the merge and uses one canonical copy passed separately
# by verilator_fifosim_multi, so per-instance copies never actually reach
# the compile and don't need renaming either. axis_infrastructure_v1_1_vl_rfs:
# a generic Xilinx vendor "reference file set" bundling many AXI-stream
# infrastructure utility modules (e.g. util_aclken_converter_wrapper) --
# content-identical across every partition (same ipshared/<hash> cache
# dir contents everywhere); renaming it caused instantiations elsewhere
# in a partition's own sources to reference a renamed module whose
# (unrenamed) declaration Verilator couldn't find via -y auto-discovery.
# Leaving duplicate-but-identical declarations in the merged file is
# harmless under Verilator (-Wno-fatal), unlike the MVAU_hls_0-style
# per-instance-divergent-ROM collisions this whole mechanism targets.


def _collect_verilog_module_decls(path):
    with open(path, errors="ignore") as f:
        text = f.read()
    return set(re.findall(r"^\s*module\s+(\w+)", text, re.MULTILINE))


def _rename_partition_verilog_sources(node_name, orig_list_path, top_module_name, out_dir):
    """Copies a partition's own all_verilog_srcs.txt file set into out_dir,
    renaming every internally-declared Verilog module (and same-partition
    references to it) with a partition-unique prefix, and returns the path
    to a new all_verilog_srcs.txt pointing at the renamed copies.

    Necessary because each partition was HLS/ipgen-synthesized completely
    independently, so many of its leaf module names (e.g. "MVAU_hls_0",
    "StreamingFIFO_rtl_0", "Q_srl") are GENERIC -- numbering restarts at 0
    for every separate per-partition build -- and collide with the same
    generic names in every other partition. Combining all partitions'
    sources (finn.util.pyverilator.prepare_stitched_ip_for_verilator
    concatenates them into a single flat Verilog source for Verilator)
    turns these into literal duplicate `module NAME ... endmodule`
    declarations in ONE compile unit; Verilator (run with -Wno-fatal)
    silently keeps only one definition and reuses it for every same-named
    instance across ALL partitions -- wiring the wrong partition's
    generated RTL (e.g. mismatched embedded ROM threshold constants) into
    other instances, surfacing as a $readmem "address beyond bounds of
    array" abort deep in an unrelated partition at simulation time.
    """

    with open(orig_list_path) as f:
        orig_files = [line.strip() for line in f if line.strip()]

    module_names = set()
    for fn in orig_files:
        if any(s in fn for s in _SKIP_RENAME_SUBSTRINGS):
            continue
        if fn.endswith(".v") or fn.endswith(".sv"):
            module_names |= _collect_verilog_module_decls(fn)
    # the partition's own top wrapper name is already globally unique
    # (CreateStitchedIP packages each partition with its own partition
    # name as ip_name) -- leave it alone so the hand-written top-level
    # netlist's instantiation of it still matches.
    module_names.discard(top_module_name)
    rename_map = {m: "%s_r_%s" % (node_name, m) for m in module_names}

    # Build ONE combined regex (alternation over all names, longest first so a
    # shorter name can't shadow a longer one that starts with it) instead of
    # looping re.sub() per name per file -- with partitions containing
    # thousands of per-channel ROM modules, the naive O(files * names)
    # approach took tens of minutes per partition; a single compiled
    # alternation pattern makes each file a single O(len(text)) pass.
    if rename_map:
        sorted_names = sorted(rename_map, key=len, reverse=True)
        combined_re = re.compile(
            r"\b(?:%s)\b" % "|".join(re.escape(n) for n in sorted_names)
        )
    else:
        combined_re = None

    os.makedirs(out_dir, exist_ok=True)
    new_files = []
    seen_basenames = set()
    for fn in orig_files:
        if any(s in fn for s in _SKIP_RENAME_SUBSTRINGS) or not (
            fn.endswith(".v") or fn.endswith(".sv")
        ):
            if "swg_pkg" in fn:
                # Vivado's OOC synth (vivadoprojgen.sh) re-globs+re-sorts all
                # sources alphabetically, discarding all_verilog_srcs.txt's
                # original (dependency-correct) order. Plain "swg_pkg.sv"
                # sorts AFTER "GenericPartition_*_swg_common*.sv" (which
                # `import swg::*;`), so Vivado elaborates the package's
                # users before the package itself and fails with
                # "'swg' is not declared". Give it a basename that sorts
                # first (leading digit) so the package is always read first;
                # the package NAME ("swg") inside is untouched, so this is
                # safe even though other files reference it by package name,
                # not filename.
                sorted_first_path = os.path.join(out_dir, "0_swg_pkg.sv")
                if not os.path.exists(sorted_first_path):
                    shutil.copy2(fn, sorted_first_path)
                new_files.append(sorted_first_path)
            else:
                new_files.append(fn)  # unchanged: shared/excluded or non-source (.dat/.vh/etc)
            continue
        with open(fn, errors="ignore") as f:
            text = f.read()
        if combined_re is not None:
            text = combined_re.sub(lambda m: rename_map[m.group(0)], text)
        # prefix with node_name (partition-unique) so basenames stay unique
        # not just across this partition's own files, but GLOBALLY across
        # all partitions -- required because Vivado's OOC-synthesis project
        # import (unlike Verilator, which takes full paths as-is) silently
        # flattens/copies added sources by basename alone, so two
        # partitions independently generating e.g.
        # "ConvolutionInputGenerator_rtl_1_wrapper.v" (per-partition node
        # numbering restarts at 0) would otherwise have one silently
        # overwrite/shadow the other, leaving dangling references to a
        # renamed module whose declaration never made it into Vivado's copy.
        base = "%s_%s" % (node_name, os.path.basename(fn))
        if base in seen_basenames:
            stem, ext = os.path.splitext(base)
            base = "%s_%d%s" % (stem, len(new_files), ext)
        seen_basenames.add(base)
        out_path = os.path.join(out_dir, base)
        with open(out_path, "w") as f:
            f.write(text)
        new_files.append(out_path)

    new_list_path = os.path.join(out_dir, "all_verilog_srcs.txt")
    with open(new_list_path, "w") as f:
        f.write("\n".join(new_files) + "\n")
    return new_list_path


def _leads_to_graph_output(tensor_name, model, graph_output_names, _seen=None):
    """True if tensor_name is a graph output, or eventually feeds one
    through a chain of non-hw nodes (Transpose/Mul/etc. downstream of the
    HW region). False if the chain dead-ends with no further consumer
    before reaching a real graph output (this happens for at least one
    genuinely-unused branch in this network, e.g. an abandoned aux head)."""

    if tensor_name in graph_output_names:
        return True
    if _seen is None:
        _seen = set()
    if tensor_name in _seen:
        return False
    _seen.add(tensor_name)
    consumers = model.find_consumers(tensor_name) or []
    for c in consumers:
        for out_name in c.output:
            if _leads_to_graph_output(out_name, model, graph_output_names, _seen):
                return True
    return False


def step_combine_partitions(model, cfg):
    """New top-level stitch: writes a plain, hand-generated Verilog top
    module ("finn_design_wrapper.v") that directly instantiates each
    partition's own already-packaged/working stitched-IP wrapper module
    (e.g. "GenericPartition_0_wrapper"), wiring adjacent partitions'
    AXI-streams together (matched by actual ONNX tensor name, since some
    partition boundaries carry more than one parallel stream -- this is
    NOT a strict single-stream linear chain), tying off any HW-boundary
    stream that doesn't lead to a real model graph output (dead
    branches can exist in the exported ONNX graph), and exposing the
    single true external input/output stream at the top level.

    This intentionally bypasses Vivado's BD/IP-Integrator (create_bd_cell
    + generate_target) entirely: an earlier attempt at using it hit a
    hard, unresolvable "Cannot upgrade to invalid target ''" error from
    Vivado when trying to (re)generate the nested child IPs of an
    already-packaged partition-level stitched IP referenced from a
    *different* Vivado project via ip_repo_paths -- a real Vivado
    limitation/bug for this specific doubly-nested cross-project IP
    scenario, not something our own tcl could fix. Since downstream
    tooling (finn.util.pyverilator.prepare_stitched_ip_for_verilator and
    finn.transformation.fpgadataflow.synth_ooc.SynthOutOfContext) only
    ever needs "vivado_stitch_proj/all_verilog_srcs.txt" +
    "wrapper_filename" metadata -- a flat list of Verilog files and a top
    module name -- and never actually requires a Vivado project/BD to
    exist, we can skip Vivado for combining altogether and just merge
    each partition's own (already proven-working) all_verilog_srcs.txt
    plus our own hand-written top wrapper file.
    """

    sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) > 0, "No partitions to combine"
    sdp_node_names = set(n.name for n in sdp_nodes)
    graph_output_names = set(o.name for o in model.graph.output)

    vivado_stitch_proj_dir = make_build_dir("combined_stitch_proj_")
    clk_ns = cfg.synth_clk_period_ns
    # NOTE: must be "finn_design"/"finn_design_wrapper" (CreateStitchedIP's
    # own default ip_name) -- finn.util.pyverilator.verilator_fifosim
    # (used by step_measure_rtlsim_performance_multi) and the generated
    # verilator_fifosim.cpp driver hardcode this exact top module/file name.
    block_name = "finn_design"

    partition_info = {}
    partition_v_src_lists = []
    rename_out_root = os.path.join(vivado_stitch_proj_dir, "renamed_src")
    for node in sdp_nodes:
        sdp_inst = getCustomOp(node)
        dataflow_model_filename = sdp_inst.get_nodeattr("model")
        kernel_model = ModelWrapper(dataflow_model_filename)

        ipstitch_path = kernel_model.get_metadata_prop("vivado_stitch_proj")
        wrapper_filename = kernel_model.get_metadata_prop("wrapper_filename")
        assert ipstitch_path is not None and wrapper_filename is not None, (
            "Partition %s has no stitched IP -- did step_build_all_partitions run?" % node.name
        )
        module_name = os.path.basename(wrapper_filename)
        if module_name.endswith(".v"):
            module_name = module_name[:-2]
        partition_info[node.name] = {
            "module_name": module_name,
            "port_widths": _parse_wrapper_ports(wrapper_filename),
        }
        renamed_list_path = _rename_partition_verilog_sources(
            node.name,
            ipstitch_path + "/all_verilog_srcs.txt",
            module_name,
            os.path.join(rename_out_root, node.name),
        )
        partition_v_src_lists.append(renamed_list_path)

    # assign a net name (and tdata width) to every tensor crossing a
    # partition-to-partition boundary, plus every true external
    # input/output boundary -- tensors whose HW-side producer/consumer
    # exists but which don't lead to a real graph output are left
    # unassigned (tensor_net) and get tied off at instantiation time.
    tensor_net = {}
    ext_inputs = []  # (netbase, width) in top-level port order
    ext_outputs = []  # (netbase, width) in top-level port order

    for node in sdp_nodes:
        pw = partition_info[node.name]["port_widths"]
        for idx, tensor_name in enumerate(node.input):
            prod = model.find_producer(tensor_name)
            if prod is None or prod.name not in sdp_node_names:
                if tensor_name in tensor_net:
                    continue
                width = pw.get("s_axis_%d_tdata" % idx, "")
                netbase = "s_axis_%d" % len(ext_inputs)
                tensor_net[tensor_name] = (netbase, width)
                ext_inputs.append((netbase, width))
        for idx, tensor_name in enumerate(node.output):
            consumers = model.find_consumers(tensor_name) or []
            sdp_consumers = [c for c in consumers if c.name in sdp_node_names]
            width = pw.get("m_axis_%d_tdata" % idx, "")
            if sdp_consumers:
                tensor_net[tensor_name] = ("net_%s_%d" % (node.name, idx), width)
            elif _leads_to_graph_output(tensor_name, model, graph_output_names):
                netbase = "m_axis_%d" % len(ext_outputs)
                tensor_net[tensor_name] = (netbase, width)
                ext_outputs.append((netbase, width))
            # else: dead branch (no consumer, or consumer chain never
            # reaches a real graph output) -- left out of tensor_net,
            # tied off below.

    assert len(ext_inputs) >= 1, "No external input stream found"
    assert len(ext_outputs) >= 1, "No external (live) output stream found"

    # --- build the plain Verilog top module ---
    lines = []
    lines.append("`timescale 1 ps / 1 ps")
    lines.append("")
    port_names = ["ap_clk", "ap_rst_n"]
    for netbase, _ in ext_inputs:
        port_names += [netbase + "_tdata", netbase + "_tvalid", netbase + "_tready"]
    for netbase, _ in ext_outputs:
        port_names += [netbase + "_tdata", netbase + "_tvalid", netbase + "_tready"]
    lines.append("module %s_wrapper (" % block_name)
    lines.append("    " + ",\n    ".join(port_names))
    lines.append(");")
    lines.append("  input ap_clk;")
    lines.append("  input ap_rst_n;")
    for netbase, width in ext_inputs:
        lines.append("  input %s%s_tdata;" % (width + " " if width else "", netbase))
        lines.append("  input %s_tvalid;" % netbase)
        lines.append("  output %s_tready;" % netbase)
    for netbase, width in ext_outputs:
        lines.append("  output %s%s_tdata;" % (width + " " if width else "", netbase))
        lines.append("  output %s_tvalid;" % netbase)
        lines.append("  input %s_tready;" % netbase)
    lines.append("")

    # internal-only nets (SDP-to-SDP boundaries) need explicit wire decls;
    # external ports are already declared above via input/output.
    ext_netbases = set(nb for nb, _ in ext_inputs) | set(nb for nb, _ in ext_outputs)
    for tensor_name, (netbase, width) in tensor_net.items():
        if netbase in ext_netbases:
            continue
        lines.append("  wire %s%s_tdata;" % (width + " " if width else "", netbase))
        lines.append("  wire %s_tvalid;" % netbase)
        lines.append("  wire %s_tready;" % netbase)
    lines.append("")

    for node in sdp_nodes:
        info = partition_info[node.name]
        conn = [".ap_clk(ap_clk)", ".ap_rst_n(ap_rst_n)"]
        for idx, tensor_name in enumerate(node.input):
            prefix = "s_axis_%d" % idx
            netbase, _ = tensor_net[tensor_name]
            conn += [
                ".%s_tdata(%s_tdata)" % (prefix, netbase),
                ".%s_tvalid(%s_tvalid)" % (prefix, netbase),
                ".%s_tready(%s_tready)" % (prefix, netbase),
            ]
        for idx, tensor_name in enumerate(node.output):
            prefix = "m_axis_%d" % idx
            if tensor_name in tensor_net:
                netbase, _ = tensor_net[tensor_name]
                conn += [
                    ".%s_tdata(%s_tdata)" % (prefix, netbase),
                    ".%s_tvalid(%s_tvalid)" % (prefix, netbase),
                    ".%s_tready(%s_tready)" % (prefix, netbase),
                ]
            else:
                # dead/unused output stream -- leave tdata/tvalid
                # unconnected, tie tready high so the partition is never
                # backpressured waiting for a reader that doesn't exist
                conn += [
                    ".%s_tdata()" % prefix,
                    ".%s_tvalid()" % prefix,
                    ".%s_tready(1'b1)" % prefix,
                ]
        lines.append(
            "  %s %s (\n    %s\n  );" % (info["module_name"], node.name, ",\n    ".join(conn))
        )
        lines.append("")

    lines.append("endmodule")

    hdl_dir = os.path.join(vivado_stitch_proj_dir, "hdl")
    os.makedirs(hdl_dir, exist_ok=True)
    wrapper_filename = os.path.join(hdl_dir, "%s_wrapper.v" % block_name)
    with open(wrapper_filename, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(
        "[step_combine_partitions] wrote hand-generated top wrapper %s "
        "(%d partitions, %d external inputs, %d external outputs)"
        % (wrapper_filename, len(sdp_nodes), len(ext_inputs), len(ext_outputs))
    )

    # merge each partition's own all_verilog_srcs.txt (real RTL source
    # paths from its own, already fully-working CreateStitchedIP run)
    # plus our own hand-written wrapper file
    merged_srcs = [wrapper_filename]
    for part_list in partition_v_src_lists:
        assert os.path.isfile(part_list), "Missing %s from partition build" % part_list
        with open(part_list) as f:
            merged_srcs.extend([line.strip() for line in f if line.strip()])
    seen = set()
    deduped_srcs = []
    for s in merged_srcs:
        if s not in seen:
            seen.add(s)
            deduped_srcs.append(s)
    v_file_list = os.path.join(vivado_stitch_proj_dir, "all_verilog_srcs.txt")
    with open(v_file_list, "w") as f:
        f.write("\n".join(deduped_srcs) + "\n")
    print(
        "[step_combine_partitions] merged %d verilog source paths from %d partitions into %s"
        % (len(deduped_srcs), len(partition_v_src_lists), v_file_list)
    )

    model.set_metadata_prop("vivado_stitch_proj", vivado_stitch_proj_dir)
    model.set_metadata_prop("wrapper_filename", wrapper_filename)
    model.set_metadata_prop("clk_ns", str(clk_ns))
    print("[step_combine_partitions] combined stitch ready: %s" % vivado_stitch_proj_dir)
    return model


def step_generate_estimate_reports_multi(model, cfg):
    """Multi-partition replacement for the generic
    build_dataflow_steps.step_generate_estimate_reports: after
    step_combine_partitions, the top-level model's nodes are all
    StreamingDataflowPartition (plus non-hw Transpose/Mul) -- not the
    actual HW nodes the generic per-node analyses expect -- so this runs
    the same analyses on each partition's own child model instead and
    merges the results into the same report filenames/format the generic
    step produces, so downstream tooling is unaffected.

    Per-layer dicts are merged with a "<partition_name>_" key prefix to
    avoid name collisions across partitions. Critical path cycles are
    summed across partitions (strict linear AXI-stream chain, so the
    combined pessimistic critical path is the sum of each partition's
    own); max_cycles/max_cycles_node_name track the single slowest node
    across all partitions.
    """

    if DataflowOutputType.ESTIMATE_REPORTS not in cfg.generate_outputs:
        return model

    report_dir = cfg.output_dir + "/report"
    os.makedirs(report_dir, exist_ok=True)

    sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")

    ops_and_params = {}
    estimate_layer_cycles = {}
    estimate_layer_resources = {}
    estimate_layer_resources_complete = {}
    total_critical_path_cycles = 0
    total_max_cycles = 0
    total_max_cycles_node_name = ""

    for node in sdp_nodes:
        prefix = node.name + "_"
        kernel_model = ModelWrapper(getCustomOp(node).get_nodeattr("model"))

        part_ops_and_params = kernel_model.analysis(op_and_param_counts)
        ops_and_params.update({prefix + k: v for k, v in part_ops_and_params.items()})

        part_layer_cycles = kernel_model.analysis(exp_cycles_per_layer)
        estimate_layer_cycles.update({prefix + k: v for k, v in part_layer_cycles.items()})

        part_layer_resources = kernel_model.analysis(
            partial(res_estimation, fpgapart=cfg._resolve_fpga_part())
        )
        estimate_layer_resources.update({prefix + k: v for k, v in part_layer_resources.items()})

        part_layer_resources_complete = kernel_model.analysis(
            partial(res_estimation_complete, fpgapart=cfg._resolve_fpga_part())
        )
        estimate_layer_resources_complete.update(
            {prefix + k: v for k, v in part_layer_resources_complete.items()}
        )

        kernel_model = kernel_model.transform(AnnotateCycles())
        part_perf = kernel_model.analysis(dataflow_performance)
        total_critical_path_cycles += part_perf["critical_path_cycles"]
        if part_perf["max_cycles"] > total_max_cycles:
            total_max_cycles = part_perf["max_cycles"]
            total_max_cycles_node_name = prefix + part_perf["max_cycles_node_name"]

    with open(report_dir + "/op_and_param_counts.json", "w") as f:
        json.dump(ops_and_params, f, indent=2)
    with open(report_dir + "/estimate_layer_cycles.json", "w") as f:
        json.dump(estimate_layer_cycles, f, indent=2)
    estimate_layer_resources["total"] = aggregate_dict_keys(estimate_layer_resources)
    with open(report_dir + "/estimate_layer_resources.json", "w") as f:
        json.dump(estimate_layer_resources, f, indent=2)
    with open(report_dir + "/estimate_layer_config_alternatives.json", "w") as f:
        json.dump(estimate_layer_resources_complete, f, indent=2)

    estimate_network_performance = {
        "critical_path_cycles": int(total_critical_path_cycles),
        "max_cycles": int(total_max_cycles),
        "max_cycles_node_name": total_max_cycles_node_name,
    }
    n_clock_cycles_per_sec = (10**9) / cfg.synth_clk_period_ns
    estimate_network_performance["estimated_throughput_fps"] = (
        n_clock_cycles_per_sec / estimate_network_performance["max_cycles"]
    )
    estimate_network_performance["estimated_latency_ns"] = (
        estimate_network_performance["critical_path_cycles"] * cfg.synth_clk_period_ns
    )
    with open(report_dir + "/estimate_network_performance.json", "w") as f:
        json.dump(estimate_network_performance, f, indent=2)

    print("[step_generate_estimate_reports_multi] reports written to %s" % report_dir)
    return model


def verilator_fifosim_multi(model, n_inputs, ishape_folded, oshape_folded, max_iters=100000000):
    """Same as finn.util.pyverilator.verilator_fifosim, but for a combined
    multi-partition model: the top-level graph's own declared input/output
    tensors are consumed/produced by non-hw Transpose/Mul nodes (software
    pre/post-processing, not part of the stitched hardware itself), so
    model.find_consumer/find_producer + getCustomOp on the top-level model
    (what the original function does) doesn't work here. Instead the
    folded input/output shapes are passed in directly (the caller derives
    them from the first partition's own first HW node and the last
    partition's own last HW node). FIFO depth-monitor logging (a
    diagnostic-only feature in the original) is omitted since correctly
    renumbering its signals across 8 separately-stitched partitions
    folded into one flat top module is fragile; core cycle/throughput/
    latency measurement (this function's actual purpose) is unaffected.
    """

    vivado_stitch_proj_dir = prepare_stitched_ip_for_verilator(model)
    verilog_header_dir = vivado_stitch_proj_dir + "/pyverilator_vh"
    build_dir = make_build_dir("verilator_fifosim_")
    fifosim_cpp_fname = os.environ["FINN_ROOT"] + "/src/finn/qnn-data/cpp/verilator_fifosim.cpp"
    with open(fifosim_cpp_fname, "r") as f:
        fifosim_cpp_template = f.read()

    template_dict = {
        "ITERS_PER_INPUT": np.prod(ishape_folded[:-1]),
        "ITERS_PER_OUTPUT": np.prod(oshape_folded[:-1]),
        "N_INPUTS": n_inputs,
        "MAX_ITERS": max_iters,
        "FIFO_DEPTH_LOGGING": "",
    }

    for key, val in template_dict.items():
        fifosim_cpp_template = fifosim_cpp_template.replace(f"@{key}@", str(val))

    with open(build_dir + "/verilator_fifosim.cpp", "w") as f:
        f.write(fifosim_cpp_template)

    which_verilator = shutil.which("verilator")
    if which_verilator is None:
        raise Exception("'verilator' executable not found")

    # add defines to make certain XPM src files work with Verilator
    xpm_args = ["-DDISABLE_XPM_ASSERTIONS", "-DOBSOLETE", "-DONESPIN", "--bbox-unsup"]
    vivado_path = os.environ["VIVADO_PATH"]
    # additional SystemVerilog modules to make XPMs work with Verilator
    xpm_memory = f"{vivado_path}/data/ip/xpm/xpm_memory/hdl/xpm_memory.sv"
    xpm_cdc = f"{vivado_path}/data/ip/xpm/xpm_cdc/hdl/xpm_cdc.sv"
    xpm_fifo = f"{vivado_path}/data/ip/xpm/xpm_fifo/hdl/xpm_fifo.sv"
    swg_pkg = os.environ["FINN_ROOT"] + "/finn-rtllib/swg/swg_pkg.sv"
    verilog_file_arg = [swg_pkg, "finn_design_wrapper.v", xpm_memory, xpm_cdc, xpm_fifo]

    verilator_args = [
        "perl",
        which_verilator,
        "-Wno-fatal",
        "-Mdir",
        build_dir,
        "-y",
        vivado_stitch_proj_dir,
        "-y",
        verilog_header_dir,
        "--CFLAGS",
        "--std=c++11",
        "-O3",
        "--x-assign",
        "fast",
        "--x-initial",
        "fast",
        "--noassert",
        "--cc",
        *verilog_file_arg,
        "--top-module",
        "finn_design_wrapper",
        "--exe",
        "verilator_fifosim.cpp",
        "--threads",
        "4",
        *xpm_args,
    ]

    proc_env = os.environ.copy()
    gcc_args = "-O3 -march=native"
    proc_env["OPT_FAST"] = gcc_args
    make_args = [
        "make",
        "-j4",
        "-C",
        build_dir,
        "-f",
        "Vfinn_design_wrapper.mk",
        "Vfinn_design_wrapper",
    ]

    with open(build_dir + "/compile.sh", "w") as f:
        f.write("#!/bin/bash \n")
        f.write("export OPT_FAST='%s'\n" % gcc_args)
        f.write(" ".join(verilator_args) + "\n")
        f.write(" ".join(make_args) + "\n")

    launch_process_helper(verilator_args, cwd=build_dir)
    launch_process_helper(make_args, proc_env=proc_env, cwd=build_dir)

    sim_launch_args = ["./Vfinn_design_wrapper"]
    launch_process_helper(sim_launch_args, cwd=build_dir)

    with open(build_dir + "/results.txt", "r") as f:
        results = f.read().strip().split("\n")
    ret_dict = {}
    for result_line in results:
        key, val = result_line.split("\t")
        ret_dict[key] = int(val)
    return ret_dict


def step_measure_rtlsim_performance_multi(model, cfg):
    """Multi-partition replacement for the generic
    build_dataflow_steps.step_measure_rtlsim_performance: the folded
    input/output shapes needed to size the C++ rtlsim driver are derived
    from the first partition's own first HW node and the last
    partition's own last HW node (loaded from their child .onnx models),
    since the combined top-level model's declared graph input/output are
    consumed/produced by non-hw Transpose/Mul nodes that aren't part of
    the stitched hardware. Measures the actual combined stitched design
    (all 8 partitions wired together), same as CreateStitchedIP's own
    rtlsim would for a single-partition build.
    """

    if DataflowOutputType.RTLSIM_PERFORMANCE not in cfg.generate_outputs:
        return model
    assert (
        DataflowOutputType.STITCHED_IP in cfg.generate_outputs
    ), "rtlsim_perf needs stitched IP"

    report_dir = cfg.output_dir + "/report"
    os.makedirs(report_dir, exist_ok=True)

    sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
    sdp_node_names = set(n.name for n in sdp_nodes)
    graph_output_names = set(o.name for o in model.graph.output)

    first_kernel = ModelWrapper(getCustomOp(sdp_nodes[0]).get_nodeattr("model"))
    first_node = first_kernel.find_consumer(first_kernel.graph.input[0].name)
    ishape_folded = getCustomOp(first_node).get_folded_input_shape()

    # find whichever SDP node/output tensor is the true external (live)
    # output stream -- not necessarily the last node's output[0]: some
    # partitions have >1 output stream and/or a dead branch that never
    # reaches a real graph output (see step_combine_partitions). Track
    # the positional INDEX (not just the tensor name): the child kernel
    # model's own internal tensor names can diverge from the parent SDP
    # node's output name strings (GiveReadableTensorNames reruns inside
    # _build_one_partition and may rename them), but partitioning always
    # preserves output list order 1:1 between an SDP node and its own
    # child model's graph.output.
    live_output_node, live_output_idx = None, None
    for node in sdp_nodes:
        for idx, tensor_name in enumerate(node.output):
            consumers = model.find_consumers(tensor_name) or []
            sdp_consumers = [c for c in consumers if c.name in sdp_node_names]
            if not sdp_consumers and _leads_to_graph_output(tensor_name, model, graph_output_names):
                live_output_node, live_output_idx = node, idx
    assert live_output_node is not None, "No live external output stream found"
    last_kernel = ModelWrapper(getCustomOp(live_output_node).get_nodeattr("model"))
    last_node = last_kernel.find_producer(last_kernel.graph.output[live_output_idx].name)
    oshape_folded = getCustomOp(last_node).get_folded_output_shape()

    rtlsim_model = deepcopy(model)
    rtlsim_bs = int(cfg.rtlsim_batch_size)
    rtlsim_perf_dict = verilator_fifosim_multi(rtlsim_model, rtlsim_bs, ishape_folded, oshape_folded)

    cycles = rtlsim_perf_dict["cycles"]
    clk_ns = float(model.get_metadata_prop("clk_ns"))
    fclk_mhz = 1 / (clk_ns * 0.001)
    runtime_s = (cycles * clk_ns) * (10**-9)
    rtlsim_perf_dict["runtime[ms]"] = runtime_s * 1000
    rtlsim_perf_dict["throughput[images/s]"] = rtlsim_bs / runtime_s
    rtlsim_perf_dict["fclk[mhz]"] = fclk_mhz

    if rtlsim_bs == 1:
        rtlsim_perf_dict["stable_throughput[images/s]"] = rtlsim_perf_dict["throughput[images/s]"]
    else:
        total_cycles = rtlsim_perf_dict["cycles"]
        latency_cycles = rtlsim_perf_dict["latency_cycles"]
        stablestate_cycles = total_cycles - latency_cycles
        stablestate_runtime_s = (stablestate_cycles * clk_ns) * (10**-9)
        rtlsim_perf_dict["stable_throughput[images/s]"] = rtlsim_bs / stablestate_runtime_s

    with open(report_dir + "/rtlsim_performance.json", "w") as f:
        json.dump(rtlsim_perf_dict, f, indent=2)

    print("[step_measure_rtlsim_performance_multi] rtlsim performance written to %s" % report_dir)
    return model


def step_out_of_context_synthesis_multi(model, cfg):
    """Multi-partition replacement for the generic
    build_dataflow_steps.step_out_of_context_synthesis: identical except
    it does not call `model.analysis(dataflow_performance)` on the
    combined top-level model afterward (that would fail the same way
    step_generate_estimate_reports did on the raw SDP/Transpose/Mul
    parent graph -- see step_generate_estimate_reports_multi). Reuses the
    "max_cycles" figure already computed there (from report/
    estimate_network_performance.json) for the estimated-fps calculation
    instead.
    """

    from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext

    if DataflowOutputType.OOC_SYNTH not in cfg.generate_outputs:
        return model
    assert DataflowOutputType.STITCHED_IP in cfg.generate_outputs, "OOC needs stitched IP"

    model = model.transform(
        SynthOutOfContext(part=cfg._resolve_fpga_part(), clk_period_ns=cfg.synth_clk_period_ns)
    )
    report_dir = cfg.output_dir + "/report"
    os.makedirs(report_dir, exist_ok=True)
    ooc_res_dict = model.get_metadata_prop("res_total_ooc_synth")
    ooc_res_dict = eval(ooc_res_dict)

    with open(report_dir + "/estimate_network_performance.json") as f:
        estimate_network_performance = json.load(f)
    n_clock_cycles_per_sec = float(ooc_res_dict["fmax_mhz"]) * (10**6)
    est_fps = n_clock_cycles_per_sec / estimate_network_performance["max_cycles"]
    ooc_res_dict["estimated_throughput_fps"] = est_fps
    with open(report_dir + "/ooc_synth_and_timing.json", "w") as f:
        json.dump(ooc_res_dict, f, indent=2)

    print("[step_out_of_context_synthesis_multi] OOC synthesis report written to %s" % report_dir)
    return model

