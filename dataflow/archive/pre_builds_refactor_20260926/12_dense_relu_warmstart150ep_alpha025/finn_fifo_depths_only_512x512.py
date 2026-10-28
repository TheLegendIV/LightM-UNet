"""FIFO-depths-only variant of finn_ooc_12_dense_relu_warmstart150ep_alpha025_
trained_rtl_mvau_8way_full_v3_nouram.py, for the 512x512 case.

Purpose: get FINN's own rtlsim-driven FIFO depth auto-sizing
(step_set_fifo_depths) for all 8 partitions WITHOUT ever invoking Vivado --
no SplitLargeFIFOs, no CreateStitchedIP, no stitched-IP build, no OOC synth.
Only Vitis HLS (step_hw_ipgen) and Verilator (rtlsim inside
step_set_fifo_depths) run. This sidesteps every failure mode we hit with the
full 512x512 build (32768 FIFO-depth ceiling, Vivado TCL cold-start race,
etc.) since those all happen strictly downstream of this point.

Reuses the v3_nouram module's folding-config bridge, MODEL_NAME/
CONV_ORDER_FILE/FOLDING_BLOCK_FILE, and per-partition step functions as-is
(imported, not copy-pasted) -- only _build_one_partition_with_folding_and_dsp
is replaced with a truncated version that stops after step_set_fifo_depths.

Run inside the FINN container, reusing the already-completed 512x512
rtl_mvau preamble (no need to re-run it):
    docker exec -e HOME=/tmp/home_dir <container> bash -c \\
        'cd /home/thelegendiv/finn/notebooks/enet && \\
         OUTDIR=finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_fifo_depths_only_512x512_$(date +%Y%m%d_%H%M%S) && \\
         python3 finn_fifo_depths_only_512x512.py \\
           finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_512x512_preamble_20260920_180600 \\
           /home/thelegendiv/finn/notebooks/enet/$OUTDIR \\
         > /tmp/finn_fifo_depths_only_512x512.log 2>&1'
"""
import concurrent.futures
import dataclasses
import csv
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.datatype import DataType  # noqa: E402
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402

from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
)

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram as v3n  # noqa: E402
sys.argv = _real_argv

FIFO_OP_TYPES = ["StreamingFIFO_rtl", "StreamingFIFO_hls", "StreamingFIFO"]


_BASE_BUILD_DIR = os.environ.get("FINN_BUILD_DIR", "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp")


def _build_one_partition_fifo_only(dataflow_model_filename, cfg, prefix, folding_config_file):
    # Same as v3n._build_one_partition_with_folding_and_dsp up through
    # step_set_fifo_depths, then stops -- no SplitLargeFIFOs/CreateStitchedIP,
    # so Vivado is never invoked for this partition.
    # NOTE: always derive from the module-level _BASE_BUILD_DIR (captured once at
    # import time), never os.environ["FINN_BUILD_DIR"] directly -- ProcessPoolExecutor
    # workers are reused across multiple partitions, and reading the (self-mutated)
    # env var back here caused build dirs to nest one level deeper per partition
    # handled by the same worker (e.g. GenericPartition_3/GenericPartition_5/...).
    part_build_dir = os.path.join(_BASE_BUILD_DIR, prefix.rstrip("_"))
    os.makedirs(part_build_dir, exist_ok=True)
    os.environ["FINN_BUILD_DIR"] = part_build_dir

    part_cfg = dataclasses.replace(cfg, folding_config_file=folding_config_file)

    kernel_model = ModelWrapper(dataflow_model_filename)
    kernel_model = step_specialize_layers(kernel_model, part_cfg)
    kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
    kernel_model = kernel_model.transform(GiveReadableTensorNames())
    kernel_model = step_target_fps_parallelization(kernel_model, part_cfg)
    kernel_model = step_apply_folding_config(kernel_model, part_cfg)
    kernel_model = v3n.step_minimize_bit_width_standalone_thresh_aware(kernel_model, part_cfg)
    kernel_model = v3n.step_fix_weight_dtype_bipolar_bug(kernel_model, part_cfg)
    kernel_model = v3n.step_force_dsp(kernel_model, part_cfg)
    kernel_model = step_hw_codegen(kernel_model, part_cfg)
    kernel_model = step_hw_ipgen(kernel_model, part_cfg)
    kernel_model = step_set_fifo_depths(kernel_model, part_cfg)

    out_path = dataflow_model_filename.replace(".onnx", "_fifo_sized.onnx")
    kernel_model.save(out_path)
    return out_path


def step_build_all_partitions_fifo_only(model, cfg, folding_config_map, max_workers=4):
    sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"

    jobs = []
    for i, sdp_node in enumerate(sdp_nodes):
        dataflow_model_filename = getCustomOp(sdp_node).get_nodeattr("model")
        prefix = sdp_node.name + "_"
        jobs.append((i, dataflow_model_filename, prefix, folding_config_map[i]))

    print(f"[fifo_only] building {len(jobs)} partitions through step_set_fifo_depths only "
          f"(parallel, max_workers={max_workers})", flush=True)
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_build_one_partition_fifo_only, fn, cfg, prefix, ffile): i
            for i, fn, prefix, ffile in jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
                print(f"[fifo_only] partition {i} done -> {results[i]}", flush=True)
            except Exception as e:  # noqa: BLE001
                results[i] = e
                print(f"[fifo_only] partition {i} FAILED: {e!r}", flush=True)

    failed = {i: e for i, e in results.items() if isinstance(e, Exception)}
    if failed:
        for i, e in failed.items():
            print(f"  FAILED partition {i}: {e!r}")
        raise RuntimeError(f"{len(failed)}/8 partitions failed: {sorted(failed.keys())}")
    return results


def dump_fifo_csv(partition_onnx_paths, out_csv):
    rows = []
    for i in sorted(partition_onnx_paths.keys()):
        part_model = ModelWrapper(partition_onnx_paths[i])
        fifo_nodes = []
        for op_type in FIFO_OP_TYPES:
            fifo_nodes += part_model.get_nodes_by_op_type(op_type)
        for n in fifo_nodes:
            inst = getCustomOp(n)
            depth = inst.get_nodeattr("depth")
            try:
                ram_style = inst.get_nodeattr("ram_style")
            except Exception:
                ram_style = ""
            bitwidth = None
            dtype = ""
            try:
                dtype = str(inst.get_nodeattr("dataType"))
                bitwidth = DataType[dtype].bitwidth()
            except Exception:
                pass
            folded_shape_str = ""
            elems_per_beat = 1
            try:
                folded_shape = inst.get_folded_output_shape()
                folded_shape_str = str(folded_shape)
                elems_per_beat = folded_shape[-1]
            except Exception:
                pass
            size_elems = elems_per_beat * depth
            size_bits = size_elems * bitwidth if bitwidth is not None else ""
            size_bytes = (size_bits / 8) if isinstance(size_bits, (int, float)) else ""
            rows.append({
                "partition": i, "node_name": n.name, "op_type": n.op_type, "depth": depth,
                "ram_style": ram_style, "dataType": dtype, "folded_shape": folded_shape_str,
                "elems_per_beat": elems_per_beat, "size_elems": size_elems,
                "bitwidth": bitwidth if bitwidth is not None else "",
                "size_bits": size_bits, "size_bytes": size_bytes,
            })
        print(f"partition {i}: {len(fifo_nodes)} FIFO nodes")

    fieldnames = ["partition", "node_name", "op_type", "depth", "ram_style", "dataType",
                  "folded_shape", "elems_per_beat", "size_elems", "bitwidth", "size_bits", "size_bytes"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} FIFO entries to {out_csv}")


def main():
    os.environ["FINN_BUILD_DIR"] = "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp"
    os.makedirs(os.environ["FINN_BUILD_DIR"], exist_ok=True)

    if len(sys.argv) < 2:
        print("Usage: finn_fifo_depths_only_512x512.py <rtl_mvau_512x512_preamble_output_dir> [explicit_output_dir]")
        sys.exit(1)
    preamble_dir = sys.argv[1]
    flat_ckpt = os.path.join(preamble_dir, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
    print(f"Preamble dir: {preamble_dir}")
    print(f"Flat 8-way-tagged checkpoint: {flat_ckpt}")
    print(f"Conv order file: {v3n.CONV_ORDER_FILE}")
    print(f"Folding block file: {v3n.FOLDING_BLOCK_FILE}")

    with open(v3n.FOLDING_BLOCK_FILE) as f:
        per_layer = json.load(f)["per_layer"]

    if len(sys.argv) >= 3:
        OUTPUT_DIR = sys.argv[2]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR = os.path.join(
            v3n.base.ENET_DIR, "finn_deployment_outputs",
            f"12_dense_relu_warmstart150ep_alpha025_fifo_depths_only_512x512_{timestamp}",
        )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"OUTPUT_DIR= {OUTPUT_DIR}", flush=True)

    cfg = dataclasses.replace(v3n.base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)
    flat_model = ModelWrapper(flat_ckpt)
    parent_model = v3n.step_create_dataflow_partition_multi(flat_model, cfg)
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    print(f"Got {len(sdp_nodes)} partitions: {[n.name for n in sdp_nodes]}")
    assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"
    v3n.validate_partition_single_output(parent_model)

    logical_by_partition = v3n.load_all_partition_logical_names(preamble_dir)
    folding_config_map = {}
    total_unmatched = 0
    for i, sdp_node in enumerate(sdp_nodes):
        conv_names, pool_names = logical_by_partition[i]
        partition_model_fn = getCustomOp(sdp_node).get_nodeattr("model")
        folding_config, n_unmatched = v3n.build_partition_folding_config(
            preamble_dir, i, sdp_node.name, partition_model_fn, conv_names, pool_names, per_layer, OUTPUT_DIR,
        )
        total_unmatched += n_unmatched
        out_path = os.path.join(OUTPUT_DIR, f"hawq_folding_config_partition{i}.json")
        with open(out_path, "w") as f:
            json.dump(folding_config, f, indent=2)
        folding_config_map[i] = out_path
        print(f"[partition {i}] saved bridged folding config ({len(folding_config) - 1} entries): {out_path}")

    print(f"\n=== Bridge summary: {total_unmatched} total unmatched logical names across all 8 partitions "
          "(expected: 0) ===\n", flush=True)

    results = step_build_all_partitions_fifo_only(parent_model, cfg, folding_config_map, max_workers=4)

    out_csv = os.path.join(OUTPUT_DIR, "fifo_depths_512x512.csv")
    dump_fifo_csv(results, out_csv)
    print("DONE. FIFO depths at:", out_csv, flush=True)
    print("OUTPUT_DIR=", OUTPUT_DIR)


if __name__ == "__main__":
    main()
