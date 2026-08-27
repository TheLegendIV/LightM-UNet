"""Verifies the new 8-way stage partitioning (stage2/3 split into 4
roughly-equal quarters, first quarter keeps down2's StreamingMaxPool even
though that breaks exact symmetry -- see assign_stage_partition_ids_8way's
docstring in finn_stage_partition.py) against the S19 model's EXISTING
step_enet_convert_to_hw.onnx checkpoint (from the already-run 5-way
partitioned build) -- reused as-is, so tidy/streamline/convert_to_hw don't
need to be re-run.

NO Vivado, NO HLS synthesis: partition_id assignment -> CreateDataflowPartition
-> per-partition specialize_layers/GiveUniqueNodeNames/target_fps_parallelization
(SAME folding config as the real 8-way build would use: target_fps=250,
mvau_wwidth_max=80, synth_clk_period_ns=10.0) -> minimize_bit_width -> FINN's
own purely-analytical estimate passes (op_and_param_counts, exp_cycles_per_layer,
res_estimation, dataflow_performance). This is meant to sanity-check partition
sizing/balance and get a rough per-partition resource estimate BEFORE
committing to another multi-day step_build_all_partitions_capped run.

Run inside the FINN container:
    docker exec <container> python3 /tmp/finn_verify_8way_stage23_split.py
"""
import json
import os
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from finn_stage_partition import assign_stage_partition_ids_8way
from finn_partition_build_steps import step_create_dataflow_partition_multi

from finn.builder.build_dataflow_config import DataflowBuildConfig
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_steps import (
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
)
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames

from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.op_and_param_counts import aggregate_dict_keys
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
# reuse the already-completed 5-way build's post-convert_to_hw checkpoint --
# identical model, only the partition_id assignment differs downstream
CHECKPOINT = os.path.join(
    ENET_DIR,
    "finn_deployment_outputs",
    "stitched_ip_partitioned_quantEnet_s19_double_mid_int8_largefifo_rtlsim_20260815_113231",
    "intermediate_models",
    "step_enet_convert_to_hw.onnx",
)

FPGA_PART = "xczu7ev-ffvc1156-2-e"
XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624, "DSP": 1728}

# same folding config the real (5-way) partitioned build uses -- kept
# identical so this estimate reflects what step_build_all_partitions would
# actually fold each partition to
cfg = DataflowBuildConfig(
    output_dir="/tmp/finn_verify_8way_stage23_split_scratch",
    mvau_wwidth_max=80,
    target_fps=250,
    synth_clk_period_ns=10.0,
    fpga_part=FPGA_PART,
    generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
)


def estimate_partition(dataflow_model_filename, prefix):
    model = ModelWrapper(dataflow_model_filename)
    model = step_specialize_layers(model, cfg)
    model = model.transform(GiveUniqueNodeNames(prefix))
    model = model.transform(GiveReadableTensorNames())
    model = step_target_fps_parallelization(model, cfg)
    model = step_apply_folding_config(model, cfg)
    model = step_minimize_bit_width(model, cfg)

    n_nodes = len(model.graph.node)
    res = model.analysis(partial(res_estimation, fpgapart=FPGA_PART))
    res["total"] = aggregate_dict_keys(res)
    model = model.transform(AnnotateCycles())
    perf = model.analysis(dataflow_performance)

    return {
        "prefix": prefix,
        "n_nodes": n_nodes,
        "LUT": res["total"].get("LUT", 0),
        "BRAM_18K": res["total"].get("BRAM_18K", 0),
        "DSP": res["total"].get("DSP", 0),
        "max_cycles": perf.get("max_cycles"),
        "critical_path_cycles": perf.get("critical_path_cycles"),
    }


if __name__ == "__main__":
    print(f"Checkpoint: {CHECKPOINT}")
    model = ModelWrapper(CHECKPOINT)
    model = assign_stage_partition_ids_8way(model)
    parent_model = step_create_dataflow_partition_multi(model, cfg)

    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    print(f"\nCreated {len(sdp_nodes)} partitions\n")

    results = []
    # keep original graph order (== partition_id order, since the parent
    # graph preserves topological node order)
    for sdp_node in sdp_nodes:
        sdp_inst = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_inst.get_nodeattr("model")
        prefix = sdp_node.name + "_"
        print(f"--- estimating {prefix} ---")
        r = estimate_partition(dataflow_model_filename, prefix)
        results.append(r)
        print(r)

    print("\n" + "=" * 100)
    header = f"{'partition':30s} {'nodes':>6} {'LUT':>8} {'LUT%':>7} {'BRAM18K':>8} {'BRAM%':>7} {'DSP':>6} {'DSP%':>6} {'max_cyc':>10}"
    print(header)
    for r in results:
        print(
            f"{r['prefix']:30s} {r['n_nodes']:6d} {int(r['LUT']):8d} {100*r['LUT']/XCZU7EV['LUT']:6.1f}% "
            f"{int(r['BRAM_18K']):8d} {100*r['BRAM_18K']/XCZU7EV['BRAM_18K']:6.1f}% "
            f"{int(r['DSP']):6d} {100*r['DSP']/XCZU7EV['DSP']:5.1f}% {r['max_cycles']:10}"
        )

    totals = {
        "n_nodes": sum(r["n_nodes"] for r in results),
        "LUT": sum(r["LUT"] for r in results),
        "BRAM_18K": sum(r["BRAM_18K"] for r in results),
        "DSP": sum(r["DSP"] for r in results),
    }
    print("-" * 100)
    print(
        f"{'TOTAL (sum of partitions)':30s} {totals['n_nodes']:6d} {int(totals['LUT']):8d} "
        f"{100*totals['LUT']/XCZU7EV['LUT']:6.1f}% {int(totals['BRAM_18K']):8d} "
        f"{100*totals['BRAM_18K']/XCZU7EV['BRAM_18K']:6.1f}% {int(totals['DSP']):6d} "
        f"{100*totals['DSP']/XCZU7EV['DSP']:5.1f}%"
    )

    out_path = os.path.join(ENET_DIR, "finn_deployment_outputs", "verify_8way_stage23_split_result.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "totals": totals}, f, indent=2)
    print(f"\nWritten: {out_path}")
