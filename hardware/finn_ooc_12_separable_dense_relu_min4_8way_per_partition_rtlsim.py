"""Real per-partition rtlsim (NOT re-synthesis -- OOC synth for all 8
partitions already completed via
finn_ooc_12_separable_dense_relu_min4_8way_per_partition_synth.py) for the
12_separable_dense_relu_min4 dummy-weight build.

Rationale: the combined-design rtlsim (step_measure_rtlsim_performance_multi,
run inside finn_ooc_12_separable_dense_relu_min4_8way_full.py) stalled --
834 inputs consumed, 0 outputs produced after the full 100M-cycle budget --
so its latency/throughput numbers are meaningless. Each partition's own
kernel model already has valid "vivado_stitch_proj"/"wrapper_filename"
CreateStitchedIP metadata from the earlier step_build_all_partitions run
(same pattern as hardware/finn_ooc_rtlsim_per_partition.py's proven,
already-used-for-s19 approach), so we can run FINN's standard single-model
step_measure_rtlsim_performance directly on each partition independently,
with no risk of the combined-design's inter-partition stitching issues.

Composition of the 8 independent results into a whole-network estimate:
  - end-to-end pipeline latency (ms) ~= SUM of each partition's own
    latency_cycles (rescaled to its own fclk_mhz -- all partitions share
    the same 10ns/100MHz target clock here, so this is a straight sum)
  - steady-state throughput (fps) ~= MIN of each partition's own
    stable_throughput[images/s] (slowest partition sets the pipeline rate,
    like the slowest station on an assembly line)
"""
import dataclasses
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from finn.builder.build_dataflow_steps import step_measure_rtlsim_performance  # noqa: E402

OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/hawq_12_sep_dense_relu_min4_dummy_8way_full_20260903_001821"
PARENT_CKPT = os.path.join(OUTPUT_DIR, "intermediate_models", "dataflow_parent_built.onnx")


def main():
    parent_model = ModelWrapper(PARENT_CKPT)
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"

    report_dir = os.path.join(OUTPUT_DIR, "report")
    os.makedirs(report_dir, exist_ok=True)

    all_results = {}
    for i, sdp_node in enumerate(sdp_nodes):
        model_path = getCustomOp(sdp_node).get_nodeattr("model")
        print(f"[per_partition_rtlsim] partition {i}: rtlsim {model_path}", flush=True)
        part_out_dir = os.path.join(OUTPUT_DIR, "per_partition_reports", sdp_node.name)
        os.makedirs(part_out_dir, exist_ok=True)
        # force_python_rtlsim=True: the default C++ verilator driver hardcodes
        # the top-module executable name "./Vfinn_design_wrapper" (only valid
        # for the combined 8-partition design); a standalone partition's top
        # module is "GenericPartition_N_wrapper", so it needs the Python
        # (pyverilator) rtlsim path instead, which discovers the top module
        # name generically.
        cfg = dataclasses.replace(
            base.cfg_stitched_ip_partitioned_8way, output_dir=part_out_dir, force_python_rtlsim=True,
        )

        part_model = ModelWrapper(model_path)
        step_measure_rtlsim_performance(part_model, cfg)

        with open(os.path.join(part_out_dir, "report", "rtlsim_performance.json")) as f:
            res = json.load(f)
        all_results[f"partition_{i}"] = res
        print(f"[per_partition_rtlsim] partition {i} result: {res}", flush=True)
        with open(os.path.join(report_dir, f"rtlsim_partition_{i}.json"), "w") as f:
            json.dump(res, f, indent=2)

    total_latency_ms = sum(
        (all_results[p]["latency_cycles"] / (all_results[p]["fclk[mhz]"] * 1000.0))
        for p in all_results
    )
    bottleneck_fps = min(all_results[p]["stable_throughput[images/s]"] for p in all_results)
    bottleneck_partition = min(
        all_results, key=lambda p: all_results[p]["stable_throughput[images/s]"]
    )

    composed = {
        "end_to_end_latency_ms_sum_of_partitions": total_latency_ms,
        "steady_state_throughput_fps_min_of_partitions": bottleneck_fps,
        "bottleneck_partition": bottleneck_partition,
        "per_partition": all_results,
    }
    with open(os.path.join(report_dir, "rtlsim_performance_per_partition_composed.json"), "w") as f:
        json.dump(composed, f, indent=2)
    print("[per_partition_rtlsim] Done. Composed report:",
          os.path.join(report_dir, "rtlsim_performance_per_partition_composed.json"))
    print(f"[per_partition_rtlsim] end-to-end latency ~= {total_latency_ms:.4f} ms, "
          f"steady-state throughput ~= {bottleneck_fps:.2f} fps (bottleneck: {bottleneck_partition})")


if __name__ == "__main__":
    main()
