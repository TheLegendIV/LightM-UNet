"""One-off TEST: run OOC synth for partition 2 AFTER forcing FIFO_MEMORY_TYPE=ultra
(URAM) on its 14 impl_style=vivado FIFOs with depth>64, to empirically validate the
FIFO-BRAM->URAM-offload estimate. Writes to a DISTINCT report filename so the
original v2_20260918 baseline result for partition 2 is not overwritten.

Usage: python3 test_partition2_fifo_uram_synth.py OUTPUT_DIR
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
    part_idx = 2
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
    print(f"[fifo_uram_test] partition {part_idx}: synthesizing {model_path}", flush=True)
    print(f"[fifo_uram_test] vivado_stitch_proj (pre-transform check via ModelWrapper metadata "
          f"not printed here; relies on already-patched /tmp/finn_dev_thelegendiv/"
          f"vivado_stitch_proj_d9ohtyc0 with 14 FIFOs forced to FIFO_MEMORY_TYPE=ultra)", flush=True)
    part_model = ModelWrapper(model_path)
    print(f"[fifo_uram_test] confirmed vivado_stitch_proj metadata = "
          f"{part_model.get_metadata_prop('vivado_stitch_proj')}", flush=True)
    part_model = part_model.transform(SynthOutOfContext(part=fpga_part, clk_period_ns=clk_period_ns))
    res = eval(part_model.get_metadata_prop("res_total_ooc_synth"))
    print(f"[fifo_uram_test] partition {part_idx} result: {res}", flush=True)
    out_path = os.path.join(report_dir, f"ooc_synth_partition_{part_idx}_fifouram_test.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[fifo_uram_test] Done, wrote {out_path}")


if __name__ == "__main__":
    main()
