"""One-off: run OOC synth for a SINGLE partition index, in parallel with the
main sequential per-partition run. Same logic as
finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_partition_synth.py
but restricted to one partition so it can be launched concurrently without
duplicating work already done/in-progress by the main run.

Usage: python3 run_single_partition_synth.py OUTPUT_DIR PARTITION_IDX
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

from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext  # noqa: E402


def main():
    OUTPUT_DIR = _real_argv[1]
    part_idx = int(_real_argv[2])
    PARENT_CKPT = os.path.join(OUTPUT_DIR, "intermediate_models", "dataflow_parent_built.onnx")
    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)
    fpga_part = cfg._resolve_fpga_part()
    clk_period_ns = cfg.synth_clk_period_ns

    parent_model = ModelWrapper(PARENT_CKPT)
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"

    report_dir = os.path.join(OUTPUT_DIR, "report")
    os.makedirs(report_dir, exist_ok=True)

    sdp_node = sdp_nodes[part_idx]
    model_path = getCustomOp(sdp_node).get_nodeattr("model")
    print(f"[single_partition_synth] partition {part_idx}: synthesizing {model_path}", flush=True)
    part_model = ModelWrapper(model_path)
    part_model = part_model.transform(SynthOutOfContext(part=fpga_part, clk_period_ns=clk_period_ns))
    res = eval(part_model.get_metadata_prop("res_total_ooc_synth"))
    print(f"[single_partition_synth] partition {part_idx} result: {res}", flush=True)
    with open(os.path.join(report_dir, f"ooc_synth_partition_{part_idx}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[single_partition_synth] Done, wrote {os.path.join(report_dir, f'ooc_synth_partition_{part_idx}.json')}")


if __name__ == "__main__":
    main()
