"""FINN full ZynqBuild (real bitstream + PYNQ driver) for JUST PARTITION 0
of the 8-way stage-partitioned S12 network (12_dense_relu_warmstart150ep_
alpha025, REAL trained ft15ep checkpoint, per-layer HAWQ bit-widths).

Motivation: the existing 8-way pipeline (finn_ooc_12_dense_relu_warmstart
150ep_alpha025_trained_8way_full.py + its per-partition OOC-synth sibling)
targets manual Vivado Block Design combination by the user (see
memories/repo/finn_12_dense_relu_alpha025_perlayer.md's "USER DECISION"
note) and OOC synthesis alone never produces a real bitstream or PYNQ
driver (no PS/DDR/DMA integration, resource/timing estimation only). This
script instead treats PARTITION 0 ALONE as a complete, standalone FINN
network and runs it through FINN's real board-integration flow (ZynqBuild:
PS/DDR/DMA wiring + full Vivado synth+impl + bitstream + PYNQ driver +
deployment package) -- same recipe as finn_zynqbuild_minimal_1bneck_int8.py,
but reusing this architecture's already-proven bridge/folding logic
(imported from finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_8way_
full.py) instead of a from-scratch derisking model.

Why a FRESH re-split instead of reusing the (now-dead) 8-way build's own
partition_0.onnx: that file was already overwritten in place by the
completed per-partition pipeline (specialize/fold/ipgen/FIFO/CreateStitchedIP
all applied) before its container (lucid_ptolemy) was lost to a WSL
auto-update (see memories/repo/finn_gotchas.md). Re-deriving partition 0's
RAW (pre-specialize) subgraph from the existing, already-completed preamble
checkpoint is deterministic and purely structural (same
finn_stage_partition.py boundary logic, no re-export/re-training involved),
so it's safe to redo into a brand-new output directory -- this script never
reads or writes anything the dead 8-way build touched.

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_zynqbuild_12_dense_relu_alpha025_partition0.py \\
        <preamble_output_dir> [explicit_output_dir]

<preamble_output_dir> example:
    /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/
    12_dense_relu_warmstart150ep_alpha025_trained_preamble_20260913_204947
"""
import dataclasses
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
sys.path.insert(0, "/home/thelegendiv/finn/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/qonnx/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/brevitas/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/pyverilator")
sys.path.insert(0, "/home/thelegendiv/finn/deps/finn-experimental")

_XILINX_BIN_DIRS = [
    "/tools/Xilinx/Vitis_HLS/2022.2/bin",
    "/tools/Xilinx/Vivado/2022.2/bin",
]
os.environ["PATH"] = os.pathsep.join(_XILINX_BIN_DIRS + [os.environ.get("PATH", "")])
os.environ.setdefault("XILINX_VIVADO", "/tools/Xilinx/Vivado/2022.2")
os.environ.setdefault("XILINX_HLS", "/tools/Xilinx/Vitis_HLS/2022.2")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames  # noqa: E402

import finn.builder.build_dataflow as build  # noqa: E402
import finn.builder.build_dataflow_config as build_cfg  # noqa: E402
from finn.builder.build_dataflow_config import DataflowBuildConfig  # noqa: E402

from finn_stage_partition import validate_partition_single_output  # noqa: E402
from finn_partition_build_steps import step_create_dataflow_partition_multi  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

# Reuse the already-proven, already-validated bridge/folding logic for this
# exact architecture (S12/12_dense_relu_warmstart150ep_alpha025, per-layer
# HAWQ) -- module import only, does not trigger that script's own main().
# NOTE: build_partition_folding_config's folding_config dict keys now carry
# the SAME partition prefix ("<sdp_node0.name>_") as the real HLS/catalog IP
# names produced below by make_step_reapply_unique_names -- both MUST use the
# identical prefix string or step_apply_folding_config silently fails to
# match any node. The prefix also keeps this partition's HLS child-IP VLNVs
# from colliding with the other 7 partitions' identically op-typed IPs if/when
# all 8 are combined into one top.bd later (see finn_gotchas.md, 2026-09-20
# MULTI-PARTITION COMBINE entry).
from finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_8way_full import (  # noqa: E402
    load_all_partition_logical_names,
    build_partition_folding_config,
    step_force_dsp,
    step_fix_weight_dtype_bipolar_bug,
    FOLDING_BLOCK_FILE,
)

BOARD = "ZCU104"  # real FINN board-file name for this chip family (xczu7ev);
                  # ZCU106 has no FINN board_files entry -- retarget the
                  # resulting Vivado project to zcu106 afterward (Apply Board
                  # Preset), see memories/repo/finn_gotchas.md.
PARTITION_IDX = 0


def make_step_reapply_unique_names(node_prefix):
    def step_reapply_unique_names(model, cfg):
        # step_specialize_layers leaves every new HLS/RTL node's .name == "",
        # which crashes HLSSynthIP's set_top with an empty top-level function
        # name -- same fix as finn_zynqbuild_minimal_1bneck_int8.py. The
        # partition-unique prefix additionally keeps this partition's HLS
        # catalog IPs from colliding with sibling partitions' identically
        # op-typed IPs once combined into one top.bd.
        return model.transform(GiveUniqueNodeNames(node_prefix))
    return step_reapply_unique_names


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <hawq_preamble_output_dir> [explicit_output_dir]")
        sys.exit(1)
    preamble_dir = sys.argv[1]
    flat_ckpt = os.path.join(preamble_dir, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
    print(f"Preamble dir: {preamble_dir}")
    print(f"Flat 8-way-tagged checkpoint: {flat_ckpt}")

    if len(sys.argv) >= 3:
        OUTPUT_DIR = sys.argv[2]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR = os.path.join(
            base.ENET_DIR, "finn_deployment_outputs",
            f"zynqbuild_12_dense_relu_warmstart150ep_alpha025_trained_partition0_{timestamp}",
        )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"OUTPUT_DIR= {OUTPUT_DIR}", flush=True)

    # ── Step A: fresh, independent re-split into 8 raw partitions (own new
    # output dir -- never touches the dead 8-way build's own files) ────────
    split_cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)
    flat_model = ModelWrapper(flat_ckpt)
    parent_model = step_create_dataflow_partition_multi(flat_model, split_cfg)
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"
    validate_partition_single_output(parent_model)
    print(f"Re-split OK: {[n.name for n in sdp_nodes]}")

    sdp_node0 = sdp_nodes[PARTITION_IDX]
    partition0_raw_fn = getCustomOp(sdp_node0).get_nodeattr("model")
    print(f"Partition {PARTITION_IDX} raw model: {partition0_raw_fn}")
    node_prefix = sdp_node0.name + "_"  # must match build_partition_folding_config's own prefix exactly

    # ── Step B: derive partition 0's bridged HAWQ folding config (identical
    # logic to the 8-way build, just for partition 0 alone) ────────────────
    with open(FOLDING_BLOCK_FILE) as f:
        per_layer = json.load(f)["per_layer"]
    logical_by_partition = load_all_partition_logical_names(preamble_dir)
    conv_names, pool_names = logical_by_partition[PARTITION_IDX]
    folding_config, n_unmatched = build_partition_folding_config(
        preamble_dir, PARTITION_IDX, sdp_node0.name, partition0_raw_fn, conv_names, pool_names, per_layer, OUTPUT_DIR,
    )
    folding_config_file = os.path.join(OUTPUT_DIR, f"hawq_folding_config_partition{PARTITION_IDX}.json")
    with open(folding_config_file, "w") as f:
        json.dump(folding_config, f, indent=2)
    print(f"Partition {PARTITION_IDX} folding config ({len(folding_config) - 1} entries): {folding_config_file}")
    print(f"Unmatched logical names: {n_unmatched} (expected 0 for this architecture)")
    if n_unmatched:
        raise RuntimeError(f"{n_unmatched} unmatched logical names -- investigate before committing to a "
                            "multi-hour Zynq build (see finn_gotchas.md's folding-bridge lessons).")

    # ── Step C: run partition 0's raw subgraph through FINN's real
    # board-integration flow, treating it as a standalone top-level model ──
    partition0_zynq_steps = [
        "step_create_dataflow_partition",
        "step_specialize_layers",
        make_step_reapply_unique_names(node_prefix),
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        step_fix_weight_dtype_bipolar_bug,
        step_force_dsp,
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
        "step_measure_rtlsim_performance",
        "step_synthesize_bitfile",
        "step_make_pynq_driver",
        "step_deployment_package",
    ]

    cfg_zynq = DataflowBuildConfig(
        output_dir          = OUTPUT_DIR,
        mvau_wwidth_max     = 80,
        target_fps          = None,   # fully unfolded (PE=SIMD=1 unless overridden by folding_config_file)
        synth_clk_period_ns = 10.0,   # 100 MHz, same conservative clock as rest of repo
        split_large_fifos   = True,
        board               = BOARD,
        shell_flow_type     = build_cfg.ShellFlowType.VIVADO_ZYNQ,
        folding_config_file = folding_config_file,
        steps               = partition0_zynq_steps,
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

    print(f"Model : {partition0_raw_fn}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Board : {BOARD}")
    print(f"Steps : {[s if isinstance(s, str) else s.__name__ for s in partition0_zynq_steps]}")
    print(flush=True)

    build.build_dataflow_cfg(partition0_raw_fn, cfg_zynq)

    print("Done. Deployment package in:", os.path.join(OUTPUT_DIR, "deploy"))
    print("OUTPUT_DIR=", OUTPUT_DIR)


if __name__ == "__main__":
    main()
