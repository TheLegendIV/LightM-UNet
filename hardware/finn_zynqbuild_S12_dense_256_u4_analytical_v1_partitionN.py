"""FINN full ZynqBuild (real PS+PL bitstream + PYNQ driver) for a SINGLE
PARTITION (index given on the command line) of the S12_dense_256_u4_analytical_v1
8-way build family -- the "alternative: single-partition ZynqBuild" track
(hardware/README.md item 6), for board bring-up/driver testing of one
partition's own real Vivado project, independent of the 8-way combine flow.

Reuses the SAME per-partition build helpers finn_s12_build.py's own 8-way
flow uses (finn_s12_build_steps.py's MILP folding bridge, bit-width/bipolar/
DSP fixups, fixed2-FIFO-insertion + MILP FIFO-depth forcing) so the result is
consistent with the validated 8-way pipeline -- it only swaps the final
steps for step_synthesize_bitfile/step_make_pynq_driver/step_deployment_package
instead of SynthOutOfContext. No partition-name-prefix fix is needed here
(that fix only matters when multiple partitions' stitched IPs share ONE flat
combined Vivado project -- this is a single, standalone partition build).

Give each partition its OWN FINN_BUILD_DIR subfolder when running more than
one of these in parallel (ZynqBuild's internal re-partitioning reuses
generic StreamingDataflowPartition_N names that would otherwise collide):
    docker exec -e HOME=/tmp/home_dir \\
        -e FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/zynqbuild_partitionN \\
        <container> bash -c 'cd /home/thelegendiv/finn/notebooks/enet && \\
        nohup python3 -u finn_zynqbuild_S12_dense_256_u4_analytical_v1_partitionN.py <partition_idx> \\
            <preamble_dir> --tag <tag> --conv-order <conv_order.json> --folding-json <layer_bits_folding_*.json> \\
            > finn_deployment_outputs/zynqbuild_partitionN.log 2>&1 &'

BOARD = "ZCU104": ZCU7EV has no FINN board_files entry -- retarget the
resulting Vivado project to the real board part afterward (see repo memory
finn_gotchas.md).
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames  # noqa: E402

import finn.builder.build_dataflow as build  # noqa: E402
import finn.builder.build_dataflow_config as build_cfg  # noqa: E402
from finn.builder.build_dataflow_config import DataflowBuildConfig  # noqa: E402
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP  # noqa: E402
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP  # noqa: E402
from finn.transformation.fpgadataflow.set_fifo_depths import SplitLargeFIFOs  # noqa: E402

from finn_stage_partition import validate_partition_single_output  # noqa: E402
from finn_partition_build_steps import step_create_dataflow_partition_multi  # noqa: E402

_real_argv, sys.argv = sys.argv, sys.argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from finn_s12_build_steps import (  # noqa: E402
    build_partition_folding_config,
    install_relaxed_stage_boundaries,
    load_partition_logical_names,
    step_fix_weight_dtype_bipolar_bug,
    step_force_dsp,
    step_force_fifo_depths_from_milp,
    step_minimize_bit_width_standalone_thresh_aware,
    step_set_fifo_depths_fixed2,
)

install_relaxed_stage_boundaries()

BOARD = "ZCU104"

_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("partition_idx", type=int)
_parser.add_argument("preamble_dir")
_parser.add_argument("--tag", required=True)
_parser.add_argument("--conv-order", required=True)
_parser.add_argument("--folding-json", required=True)
_parser.add_argument("--output-dir", default=None)
_args = _parser.parse_args()


def step_reapply_unique_names(model, cfg):
    # step_specialize_layers leaves every new HLS/RTL node's .name == "", which
    # crashes HLSSynthIP's set_top with an empty top-level function name.
    return model.transform(GiveUniqueNodeNames())


def main():
    idx = _args.partition_idx
    output_dir = _args.output_dir or os.path.join(
        base.ENET_DIR, "finn_deployment_outputs",
        f"zynqbuild_{_args.tag}_partition{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Partition idx: {idx}\nPreamble dir: {_args.preamble_dir}\nOUTPUT_DIR: {output_dir}", flush=True)

    flat_ckpt = os.path.join(_args.preamble_dir, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
    parent_model = step_create_dataflow_partition_multi(ModelWrapper(flat_ckpt), base.cfg_stitched_ip_partitioned_8way)
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"
    validate_partition_single_output(parent_model)

    sdp_node = sdp_nodes[idx]
    partition_raw_fn = getCustomOp(sdp_node).get_nodeattr("model")
    print(f"Partition {idx} raw model: {partition_raw_fn}")

    with open(_args.folding_json) as f:
        folding_block = json.load(f)
    per_layer = folding_block["per_layer"]
    extra_nodes = folding_block.get("extra_nodes")
    inter_block_fifos = folding_block.get("inter_block_fifos")
    intra_block_fifos = folding_block.get("intra_block_fifos")
    logical = load_partition_logical_names(_args.preamble_dir, _args.conv_order)

    folding_config, fifo_plan = build_partition_folding_config(
        partition_raw_fn, sdp_node.name, logical[idx][0], per_layer, base.cfg_stitched_ip_partitioned_8way,
        tag=f"p{idx}", extra_nodes=extra_nodes, conv_order_file=_args.conv_order,
        inter_block_fifos=inter_block_fifos, intra_block_fifos=intra_block_fifos,
    )
    folding_config_file = os.path.join(output_dir, f"hawq_folding_config_partition{idx}.json")
    with open(folding_config_file, "w") as f:
        json.dump(folding_config, f, indent=2)
    print(f"Partition {idx} folding config ({len(folding_config) - 1} entries): {folding_config_file}")

    def _step_force_fifo_from_milp(model, cfg):
        if fifo_plan is None:
            print(f"[partition {idx}] no FIFO plan (folding json had no inter/intra_block_fifos) "
                  "-- leaving FINN's own fixed2 depths")
            return model
        model, report = step_force_fifo_depths_from_milp(model, fifo_plan)
        with open(os.path.join(output_dir, f"fifo_force_report_partition_{idx}.json"), "w") as f:
            json.dump(report, f, indent=2)
        return model

    def _step_finalize_ipgen(model, cfg):
        # FIFO depth-forcing above inserts/resizes StreamingFIFO_rtl nodes that never went
        # through step_hw_ipgen -- without this they have no ip_path and CreateStitchedIP's
        # assert os.path.isdir(ip_dir_value) crashes (see finn_s12_build.py's identical step).
        model = model.transform(SplitLargeFIFOs())
        model = model.transform(GiveUniqueNodeNames())
        fpga_part, clk = cfg._resolve_fpga_part(), cfg.synth_clk_period_ns
        model = model.transform(PrepareIP(fpga_part, clk))
        model = model.transform(HLSSynthIP())
        return model

    zynq_steps = [
        "step_create_dataflow_partition",
        "step_specialize_layers",
        step_reapply_unique_names,
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        step_minimize_bit_width_standalone_thresh_aware,
        step_fix_weight_dtype_bipolar_bug,
        step_force_dsp,
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        step_set_fifo_depths_fixed2,
        _step_force_fifo_from_milp,
        _step_finalize_ipgen,
        "step_create_stitched_ip",
        "step_measure_rtlsim_performance",
        "step_synthesize_bitfile",
        "step_make_pynq_driver",
        "step_deployment_package",
    ]

    base_cfg = base.cfg_stitched_ip_partitioned_8way
    cfg_zynq = DataflowBuildConfig(
        output_dir          = output_dir,
        mvau_wwidth_max     = base_cfg.mvau_wwidth_max,
        target_fps          = base_cfg.target_fps,
        synth_clk_period_ns = base_cfg.synth_clk_period_ns,
        board               = BOARD,
        shell_flow_type     = build_cfg.ShellFlowType.VIVADO_ZYNQ,
        folding_config_file = folding_config_file,
        steps               = zynq_steps,
        generate_outputs    = [
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
        save_intermediate_models = True,
    )

    print(f"Model : {partition_raw_fn}\nOutput: {output_dir}\nBoard : {BOARD}", flush=True)
    build.build_dataflow_cfg(partition_raw_fn, cfg_zynq)


if __name__ == "__main__":
    main()
