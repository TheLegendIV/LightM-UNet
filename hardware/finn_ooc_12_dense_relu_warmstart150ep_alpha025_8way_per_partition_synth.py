"""Per-partition OOC synthesis for the 12_dense_relu_warmstart150ep_alpha025
8-way trained build -- same proven workaround already used for the
separable-family/w16/w20/26_9_w24 builds (see
`/memories/repo/finn_gotchas.md`'s "THIRD distinct Verilog/Tcl merge bug"
entry). Run FINN's own unmodified SynthOutOfContext transform on EACH
partition's OWN model independently instead of the combined-design
step_out_of_context_synthesis_multi path.

Byte-for-byte copy of
finn_ooc_12_separable_dense_relu_min4_8way_per_partition_synth.py (fully
generic -- no architecture-specific constants beyond DEFAULT_OUTPUT_DIR's
fallback comment) with DEFAULT_OUTPUT_DIR updated for this family.

OUTPUT_DIR is taken from sys.argv[1] (the SAME timestamped output_dir the
corresponding *_8way_full.py script used) so it can be `&&`-chained right
after that script in one shell command.
"""
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext  # noqa: E402

DEFAULT_OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_8way_full"


def main():
    if len(_real_argv) < 2:
        print(f"No OUTPUT_DIR given, falling back to DEFAULT_OUTPUT_DIR: {DEFAULT_OUTPUT_DIR}")
    OUTPUT_DIR = _real_argv[1] if len(_real_argv) >= 2 else DEFAULT_OUTPUT_DIR
    PARENT_CKPT = os.path.join(OUTPUT_DIR, "intermediate_models", "dataflow_parent_built.onnx")
    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)
    fpga_part = cfg._resolve_fpga_part()
    clk_period_ns = cfg.synth_clk_period_ns

    parent_model = ModelWrapper(PARENT_CKPT)
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"

    report_dir = os.path.join(OUTPUT_DIR, "report")
    os.makedirs(report_dir, exist_ok=True)

    # If a separate early/out-of-band OOC synth run (e.g. a manual head start
    # on already-stitched partitions) is still in flight, wait for it to
    # finish before starting our own sequential loop -- avoids launching a
    # DUPLICATE concurrent SynthOutOfContext on the same partition and avoids
    # ending up with all 8 partitions' OOC synth running at once.
    in_progress_marker = os.path.join(report_dir, ".early_ooc_synth_in_progress")
    waited = False
    while os.path.exists(in_progress_marker):
        if not waited:
            print(f"[per_partition_synth] waiting for early OOC synth run to finish "
                  f"({in_progress_marker} present)...", flush=True)
            waited = True
        time.sleep(30)
    if waited:
        print("[per_partition_synth] early OOC synth run finished, proceeding", flush=True)

    all_results = {}
    for i, sdp_node in enumerate(sdp_nodes):
        cached_report_path = os.path.join(report_dir, f"ooc_synth_partition_{i}.json")
        if os.path.exists(cached_report_path):
            with open(cached_report_path) as f:
                res = json.load(f)
            all_results[f"partition_{i}"] = res
            print(f"[per_partition_synth] partition {i}: reusing pre-existing report {cached_report_path}", flush=True)
            continue
        model_path = getCustomOp(sdp_node).get_nodeattr("model")
        print(f"[per_partition_synth] partition {i}: synthesizing {model_path}", flush=True)
        part_model = ModelWrapper(model_path)
        part_model = part_model.transform(SynthOutOfContext(part=fpga_part, clk_period_ns=clk_period_ns))
        res = eval(part_model.get_metadata_prop("res_total_ooc_synth"))
        all_results[f"partition_{i}"] = res
        print(f"[per_partition_synth] partition {i} result: {res}", flush=True)
        with open(cached_report_path, "w") as f:
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
