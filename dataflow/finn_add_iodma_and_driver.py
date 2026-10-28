"""Reusable, build-agnostic post-processing script: given an already-completed
8-way partitioned FINN OOC build's output dir (containing
intermediate_models/dataflow_parent_built.onnx with N StreamingDataflowPartition
nodes), adds a real input IODMA_hls IP to the FIRST partition and a real output
IODMA_hls IP to the LAST partition, then generates the PYNQ Python driver --
all WITHOUT re-running CreateStitchedIP/re-stitching either partition and
WITHOUT touching the original build's files (works on copies only).

Why no re-stitch is needed:
  - MakePYNQDriver only reads ONNX graph structure/nodeattrs (IODMA_hls node
    presence, burstMode, folded shapes, instance_name) -- it never looks at
    Verilog/IP-XACT, so InsertIODMA (a pure graph edit) is enough for driver
    generation.
  - The actual hardware only needs the new IODMA node to have a real
    HLSSynthIP-generated IP-XACT component (its own standalone IP); it can be
    wired directly into a hand-built Vivado Block Design next to each
    partition's own existing (untouched) stitched IP -- no need to fold it
    into that wrapper via CreateStitchedIP.
  - PrepareIP/HLSSynthIP both skip nodes that already have a valid
    ipgen_path/code_gen_dir_ipgen, so re-running them on an already-built
    partition only synthesizes the one new IODMA node.

Usage (inside the FINN container):
    docker cp hardware/finn_add_iodma_and_driver.py <container>:/home/thelegendiv/finn/notebooks/enet/
    docker exec -e HOME=/tmp/home_dir <container> bash -c \\
        "source /tools/Xilinx/Vivado/2022.2/settings64.sh && \\
         cd /home/thelegendiv/finn/notebooks/enet && \\
         python3 finn_add_iodma_and_driver.py <ooc_build_output_dir> [dest_dir]"

<ooc_build_output_dir> is the OOC build script's own OUTPUT_DIR (the one
printed as "OUTPUT_DIR= ..." by e.g. finn_ooc_..._8way_full_v4_256x256.py),
which must contain intermediate_models/dataflow_parent_built.onnx. This file
is produced right before step_combine_partitions runs, so it exists even if
the build later failed/crashed in a later step (e.g. rtlsim).

Output (under dest_dir, default <ooc_build_output_dir>/iodma_driver_<ts>/):
  kernel_models/<partition0_name>_with_input_iodma.onnx
  kernel_models/<partition_last_name>_with_output_iodma.onnx
  dataflow_parent_with_iodma.onnx        (accurate HW topology, incl. any
                                           non-HW pre/post-processing nodes)
  dataflow_parent_for_driver_gen.onnx     (same, but with non-HW pre/post
                                           nodes at the true graph boundary
                                           stripped -- MakePYNQDriver requires
                                           the graph input/output to be
                                           consumed/produced directly by a
                                           StreamingDataflowPartition node)
  preprocessing_notes.json                (details of any stripped nodes --
                                           replicate these in host-side code
                                           around the generated driver)
  driver/  (driver.py, driver_base.py, validate.py, runtime_weights/)
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402

from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA  # noqa: E402
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP  # noqa: E402
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP  # noqa: E402
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver  # noqa: E402


def add_iodma_to_partition(model_path, insert_input, insert_output, fpga_part, clk_ns,
                            max_intfwidth, insert_extmemw, label):
    model = ModelWrapper(model_path)
    orig_names = {n.name for n in model.graph.node}
    model = model.transform(
        InsertIODMA(
            max_intfwidth=max_intfwidth,
            insert_input=insert_input,
            insert_output=insert_output,
            insert_extmemw=insert_extmemw,
        )
    )
    # InsertIODMA leaves the new node(s) with name="" -- name only those, via
    # GiveUniqueNodeNames() would strip the existing GenericPartition_N_ prefix
    # from EVERY node, reintroducing the exact cross-partition name collisions
    # (StreamingFIFO_rtl_0 etc. reused in every partition) that prefix avoids.
    unnamed = [n for n in model.graph.node if n.name not in orig_names]
    for i, n in enumerate(unnamed):
        assert n.op_type == "IODMA_hls", f"unexpected new unnamed node: {n.op_type}"
        n.name = f"IODMA_hls_{i}"
    model = model.transform(PrepareIP(fpga_part, clk_ns))
    model = model.transform(HLSSynthIP())
    model.save(model_path)

    iodma_nodes = [n for n in model.graph.node if n.op_type == "IODMA_hls"]
    for n in iodma_nodes:
        inst = getCustomOp(n)
        print(f"[{label}] {n.name}: direction={inst.get_nodeattr('direction')} "
              f"ipgen_path={inst.get_nodeattr('ipgen_path')}")
    return model


def _describe_node(model, node):
    desc = {"op_type": node.op_type, "name": node.name}
    for a in node.attribute:
        if a.name == "perm":
            desc["perm"] = list(a.ints)
    for inp in node.input:
        init = model.get_initializer(inp)
        if init is not None:
            desc.setdefault("initializers", {})[inp] = {
                "shape": list(init.shape),
                "values": init.flatten().tolist() if init.size <= 64 else f"<{init.size} values, not inlined>",
            }
    return desc


def strip_non_hw_boundary_ops(model):
    """MakePYNQDriver requires model.graph.input/.output to be directly
    consumed/produced by a StreamingDataflowPartition node. Some exports
    leave plain ONNX pre/post-processing (layout Transpose, output-scale Mul,
    etc.) outside the partitioned HW region. Removes any such chain at the
    true graph boundary (rewiring the SDP node directly to the graph
    input/output tensor) and returns (model, removed_input_chain,
    removed_output_chain) for reporting -- these need to be replicated as
    host-side pre/post-processing around the generated driver."""

    removed_in = []
    for graph_in in list(model.graph.input):
        consumer = model.find_consumer(graph_in.name)
        chain = []
        while consumer is not None and consumer.op_type != "StreamingDataflowPartition":
            chain.append(consumer)
            consumer = model.find_consumer(consumer.output[0])
        if chain:
            sdp_node = consumer
            old_tensor = chain[-1].output[0]
            idx = list(sdp_node.input).index(old_tensor)
            sdp_node.input[idx] = graph_in.name
            for n in chain:
                removed_in.append(_describe_node(model, n))
                model.graph.node.remove(n)

    removed_out = []
    for graph_out in list(model.graph.output):
        producer = model.find_producer(graph_out.name)
        chain = []
        while producer is not None and producer.op_type != "StreamingDataflowPartition":
            chain.append(producer)
            producer = model.find_producer(producer.input[0])
        if chain:
            sdp_node = producer
            old_tensor = chain[-1].input[0]
            idx = list(sdp_node.output).index(old_tensor)
            sdp_node.output[idx] = graph_out.name
            for n in chain:
                removed_out.append(_describe_node(model, n))
                model.graph.node.remove(n)

    return model, removed_in, removed_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ooc_output_dir", help="completed OOC build's OUTPUT_DIR (contains intermediate_models/dataflow_parent_built.onnx)")
    ap.add_argument("dest_dir", nargs="?", default=None)
    ap.add_argument("--fpga-part", default="xczu7ev-ffvc1156-2-e")
    ap.add_argument("--clk-ns", type=float, default=10.0)
    ap.add_argument("--max-intfwidth", type=int, default=32)
    ap.add_argument("--platform", default="zynq-iodma")
    ap.add_argument("--insert-extmemw", action="store_true",
                     help="also add weight-DMA IODMAs for any external-mem-mode MVAU/VVAU nodes (off by default)")
    ap.add_argument("--partition0-idx", type=int, default=0, help="index of the partition to get the input IODMA")
    ap.add_argument("--partition-last-idx", type=int, default=-1, help="index of the partition to get the output IODMA")
    ap.add_argument("--original-build-dir", default=None,
                     help="original build's FINN_BUILD_DIR root (e.g. "
                          ".../finn_build_tmp/S12_dense_nn_upsample_256_w8_16_v4). "
                          "When given, the new IODMA node's ipgen scratch output is written "
                          "directly under <original-build-dir>/<partition-name>/ (alongside that "
                          "partition's existing per-node ipgen dirs) instead of a fresh temp dir. "
                          "Only adds a new subfolder there; never touches existing partition files.")
    args = ap.parse_args()
    if args.original_build_dir is not None:
        args.original_build_dir = os.path.abspath(args.original_build_dir)
    # FINN's ipgen/HLS subprocess launches assume FINN_BUILD_DIR (and everything
    # derived from it) is absolute -- a relative ooc_output_dir silently breaks
    # ipgen_singlenode_code()'s cwd-relative subprocess calls.
    args.ooc_output_dir = os.path.abspath(args.ooc_output_dir)
    if args.dest_dir is not None:
        args.dest_dir = os.path.abspath(args.dest_dir)

    parent_ckpt = os.path.join(args.ooc_output_dir, "intermediate_models", "dataflow_parent_built.onnx")
    assert os.path.isfile(parent_ckpt), f"not found: {parent_ckpt} -- did the OOC build get at least as far as building all partitions?"

    dest_dir = args.dest_dir or os.path.join(
        args.ooc_output_dir, f"iodma_driver_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    kernel_models_dir = os.path.join(dest_dir, "kernel_models")
    os.makedirs(kernel_models_dir, exist_ok=True)
    fallback_build_dir = os.path.join(dest_dir, "finn_build_tmp")
    os.makedirs(fallback_build_dir, exist_ok=True)

    parent_model = ModelWrapper(parent_ckpt)
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) >= 2, f"expected >=2 partitions, got {len(sdp_nodes)}"
    p0_node = sdp_nodes[args.partition0_idx]
    p_last_node = sdp_nodes[args.partition_last_idx]
    print(f"Input IODMA -> {p0_node.name} (idx {args.partition0_idx})")
    print(f"Output IODMA -> {p_last_node.name} (idx {args.partition_last_idx})")

    fn0 = getCustomOp(p0_node).get_nodeattr("model")
    fn_last = getCustomOp(p_last_node).get_nodeattr("model")

    copy0 = os.path.join(kernel_models_dir, f"{p0_node.name}_with_input_iodma.onnx")
    copy_last = os.path.join(kernel_models_dir, f"{p_last_node.name}_with_output_iodma.onnx")
    shutil.copy(fn0, copy0)
    shutil.copy(fn_last, copy_last)

    def _build_dir_for(partition_name):
        if args.original_build_dir is not None:
            d = os.path.join(args.original_build_dir, partition_name)
            os.makedirs(d, exist_ok=True)
            return d
        return fallback_build_dir

    os.environ["FINN_BUILD_DIR"] = _build_dir_for(p0_node.name)
    print(f"FINN_BUILD_DIR ({p0_node.name}) = {os.environ['FINN_BUILD_DIR']}")
    add_iodma_to_partition(copy0, insert_input=True, insert_output=False,
                            fpga_part=args.fpga_part, clk_ns=args.clk_ns,
                            max_intfwidth=args.max_intfwidth, insert_extmemw=args.insert_extmemw,
                            label=f"{p0_node.name}/input")

    os.environ["FINN_BUILD_DIR"] = _build_dir_for(p_last_node.name)
    print(f"FINN_BUILD_DIR ({p_last_node.name}) = {os.environ['FINN_BUILD_DIR']}")
    add_iodma_to_partition(copy_last, insert_input=False, insert_output=True,
                            fpga_part=args.fpga_part, clk_ns=args.clk_ns,
                            max_intfwidth=args.max_intfwidth, insert_extmemw=args.insert_extmemw,
                            label=f"{p_last_node.name}/output")

    # Fresh reload -- point just these two partitions' "model" nodeattr at the
    # IODMA-augmented copies, leave every other partition/original file untouched.
    parent_with_iodma = ModelWrapper(parent_ckpt)
    sdp_nodes2 = parent_with_iodma.get_nodes_by_op_type("StreamingDataflowPartition")
    getCustomOp(sdp_nodes2[args.partition0_idx]).set_nodeattr("model", copy0)
    getCustomOp(sdp_nodes2[args.partition_last_idx]).set_nodeattr("model", copy_last)
    parent_with_iodma_ckpt = os.path.join(dest_dir, "dataflow_parent_with_iodma.onnx")
    parent_with_iodma.save(parent_with_iodma_ckpt)

    driver_model = ModelWrapper(parent_with_iodma_ckpt)
    driver_model, removed_in, removed_out = strip_non_hw_boundary_ops(driver_model)
    if removed_in or removed_out:
        print(f"\nStripped non-HW boundary nodes for driver generation "
              f"(replicate these as host-side pre/post-processing around the driver):")
        print(f"  input side (before accelerator call): {[n['name'] for n in removed_in]}")
        print(f"  output side (after accelerator call): {[n['name'] for n in removed_out]}")
    driver_gen_ckpt = os.path.join(dest_dir, "dataflow_parent_for_driver_gen.onnx")
    driver_model.save(driver_gen_ckpt)
    notes_path = os.path.join(dest_dir, "preprocessing_notes.json")
    with open(notes_path, "w") as f:
        json.dump({"stripped_input_side_pre_accel": removed_in,
                   "stripped_output_side_post_accel": removed_out}, f, indent=2)

    driver_model = driver_model.transform(MakePYNQDriver(args.platform))
    driver_dir = os.path.join(dest_dir, "driver")
    shutil.copytree(driver_model.get_metadata_prop("pynq_driver_dir"), driver_dir, dirs_exist_ok=True)

    print(f"\nDone. dest_dir={dest_dir}")
    print(f"  parent model (2 partitions repointed): {parent_with_iodma_ckpt}")
    print(f"  driver-gen model (boundary ops stripped): {driver_gen_ckpt}")
    print(f"  preprocessing notes: {notes_path}")
    print(f"  driver: {driver_dir}")


if __name__ == "__main__":
    main()
