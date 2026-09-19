"""Kick off OOC synthesis EARLY for specific partitions of the
12_dense_relu_warmstart150ep_alpha025 8-way trained (v3) build whose stitched
IP is already finished, without waiting for the full
finn_ooc_..._8way_full_v3.py run (all 8 partitions + combine/estimate/rtlsim)
to exit.

Reads partition model paths from `intermediate_models/dataflow_parent.onnx`
(NOT `dataflow_parent_built.onnx` -- the SDP nodes' "model" nodeattrs are set
once during initial partitioning and are identical in both checkpoints; only
each partition file's on-disk CONTENTS get overwritten in-place by
`kernel_model.save()` once that partition's build finishes). This lets us
target already-finished partitions before the parent build script itself has
produced `dataflow_parent_built.onnx`.

Writes results to the SAME `report/ooc_synth_partition_{i}.json` filenames
the real `finn_ooc_..._8way_per_partition_synth.py` script uses, so that
script (patched to skip partitions with a pre-existing report) picks up
these results instead of re-synthesizing them later.

Usage:
    python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_early_partition_synth.py \
        OUTPUT_DIR 4,5,6,7
"""
import dataclasses
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext  # noqa: E402


def _synth_one_partition(output_dir, partition_idx, model_path, fpga_part, clk_period_ns):
    print(f"[early_partition_synth] partition {partition_idx}: synthesizing {model_path}", flush=True)
    part_model = ModelWrapper(model_path)
    part_model = part_model.transform(SynthOutOfContext(part=fpga_part, clk_period_ns=clk_period_ns))
    res = eval(part_model.get_metadata_prop("res_total_ooc_synth"))
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, f"ooc_synth_partition_{partition_idx}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[early_partition_synth] partition {partition_idx} result: {res}", flush=True)
    return partition_idx, res


def main():
    assert len(_real_argv) >= 3, "usage: early_partition_synth.py OUTPUT_DIR idx1,idx2,..."
    output_dir = _real_argv[1]
    partition_indices = [int(x) for x in _real_argv[2].split(",")]

    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=output_dir)
    fpga_part = cfg._resolve_fpga_part()
    clk_period_ns = cfg.synth_clk_period_ns

    parent_model = ModelWrapper(os.path.join(output_dir, "intermediate_models", "dataflow_parent.onnx"))
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"

    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    # Marker so the chained finn_ooc_..._8way_per_partition_synth.py script
    # (which runs sequentially once the main build finishes) waits for us
    # instead of launching a duplicate concurrent synth on the same
    # partition -- see that script's own wait-loop for the other half.
    in_progress_marker = os.path.join(report_dir, ".early_ooc_synth_in_progress")
    with open(in_progress_marker, "w") as f:
        f.write(f"pid={os.getpid()} partitions={partition_indices}\n")

    try:
        jobs = {}
        with ProcessPoolExecutor(max_workers=len(partition_indices)) as ex:
            for i in partition_indices:
                report_path = os.path.join(report_dir, f"ooc_synth_partition_{i}.json")
                if os.path.exists(report_path):
                    print(f"[early_partition_synth] partition {i}: report already exists, skipping: {report_path}")
                    continue
                model_path = getCustomOp(sdp_nodes[i]).get_nodeattr("model")
                fut = ex.submit(_synth_one_partition, output_dir, i, model_path, fpga_part, clk_period_ns)
                jobs[fut] = i
            for fut in as_completed(jobs):
                i = jobs[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"[early_partition_synth] partition {i} FAILED: {e}", flush=True)
                    raise
    finally:
        if os.path.exists(in_progress_marker):
            os.remove(in_progress_marker)

    print("[early_partition_synth] Done with requested partitions:", partition_indices)


if __name__ == "__main__":
    main()
