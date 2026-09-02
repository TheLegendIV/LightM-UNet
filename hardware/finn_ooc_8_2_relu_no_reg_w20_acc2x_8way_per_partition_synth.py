"""Per-partition OOC synthesis: the combined-design OOC synth
(step_out_of_context_synthesis_multi, which merges all 8 partitions into one
flat "finn_design_wrapper" top and runs SynthOutOfContext on that) hit the
THIRD distinct Verilog/Tcl merge bug across sessions on 2026-09-01: `'package'
is an unknown type` for the merged `0_swg_pkg.sv` (the vendored
`vivadoprojgen.sh`'s `is_global_include` workaround sets file_type="Verilog
Header" on that file, which makes Vivado refuse to parse its `package
...endpackage`/`logic` SystemVerilog syntax). All 8 partitions already have
valid, individually-verified, fully-built stitched IP (each went through its
own successful CreateStitchedIP, and step_combine_partitions/estimate-reports/
rtlsim-performance all completed cleanly). Run FINN's own unmodified
SynthOutOfContext transform on EACH partition's OWN model independently
instead -- this is the standard, proven, un-modified FINN path (see
finn_gotchas memory note on multi-partition builds) and completely avoids the
combine-merge bug class.
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

OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/hawq_8_2_w20_acc2x_dummy_8way_full_20260901_135031"
PARENT_CKPT = os.path.join(OUTPUT_DIR, "intermediate_models", "dataflow_parent.onnx")


def main():
    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)
    fpga_part = cfg._resolve_fpga_part()
    clk_period_ns = cfg.synth_clk_period_ns

    parent_model = ModelWrapper(PARENT_CKPT)
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"

    report_dir = os.path.join(OUTPUT_DIR, "report")
    os.makedirs(report_dir, exist_ok=True)

    all_results = {}
    for i, sdp_node in enumerate(sdp_nodes):
        model_path = getCustomOp(sdp_node).get_nodeattr("model")
        print(f"[per_partition_synth] partition {i}: synthesizing {model_path}", flush=True)
        part_model = ModelWrapper(model_path)
        part_model = part_model.transform(SynthOutOfContext(part=fpga_part, clk_period_ns=clk_period_ns))
        res = eval(part_model.get_metadata_prop("res_total_ooc_synth"))
        all_results[f"partition_{i}"] = res
        print(f"[per_partition_synth] partition {i} result: {res}", flush=True)
        with open(os.path.join(report_dir, f"ooc_synth_partition_{i}.json"), "w") as f:
            json.dump(res, f, indent=2)

    # aggregate: sum additive resource counts, report min Fmax as the
    # overall achievable clock across all 8 independent OOC kernels.
    numeric_keys = set()
    for res in all_results.values():
        for k, v in res.items():
            try:
                float(v)
                numeric_keys.add(k)
            except (TypeError, ValueError):
                pass
    aggregate = {}
    for k in numeric_keys:
        vals = [float(all_results[p][k]) for p in all_results if k in all_results[p]]
        if k.lower().startswith("fmax") or "period" in k.lower():
            aggregate[k + "_min_across_partitions"] = min(vals)
        else:
            aggregate[k + "_sum"] = sum(vals)
    all_results["aggregate"] = aggregate

    with open(os.path.join(report_dir, "ooc_synth_and_timing_per_partition.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("[per_partition_synth] Done. Combined report:",
          os.path.join(report_dir, "ooc_synth_and_timing_per_partition.json"))


if __name__ == "__main__":
    main()
