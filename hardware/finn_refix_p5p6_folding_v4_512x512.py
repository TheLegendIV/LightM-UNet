"""One-off retroactive fix for the already-crashed
finn_ooc_12_dense_relu_warmstart150ep_alpha05_trained_rtl_mvau_8way_full_v4_512x512
build (20260923_223139): partitions 5 and 6 were folded with the VVAU-PE
scaling / parallel_window bug (see that script's SWU_OP_TYPES comment,
corrected 2026-09-24). This regenerates ONLY partitions 5 and 6's
pre-FIFO-autosizing graphs with the fixed build_partition_folding_config and
overwrites ONLY their real supported_op_partitions/partition_{5,6}.onnx +
hawq_folding_config_partition{5,6}.json -- partitions 0,1,2,3,4,7's real
checkpoints (already successfully built+OOC-synthesized) are NEVER touched:
the 8-way re-split runs against a scratch output dir, not the real one.

Does NOT run step_hw_codegen/step_hw_ipgen/step_set_fifo_depths/OOC synth --
stops right after step_apply_folding_config, matching "before FIFO
autosizing". Resuming the rest of the pipeline for partitions 5/6 is a
separate, deliberate follow-up (long-running Vivado step), not part of this
script.

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> bash -c \\
        'cd /home/thelegendiv/finn/notebooks/enet && python3 finn_refix_p5p6_folding_v4_512x512.py'
"""
import dataclasses
import json
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_ooc_12_dense_relu_warmstart150ep_alpha05_trained_rtl_mvau_8way_full_v4_512x512 as buildmod  # noqa: E402
sys.argv = _real_argv

from finn_partition_build_steps import step_create_dataflow_partition_multi  # noqa: E402
from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
)

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
PREAMBLE_DIR = os.path.join(
    ENET_DIR, "finn_deployment_outputs",
    "12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_512x512_preamble_20260920_180600",
)
REAL_OUTPUT_DIR = os.path.join(
    ENET_DIR, "finn_deployment_outputs",
    "12_dense_relu_warmstart150ep_alpha05_trained_rtl_mvau_8way_full_v4_512x512_20260923_223139",
)
SCRATCH_DIR = os.path.join(ENET_DIR, "finn_build_tmp", "_refix_p5p6_scratch")
TARGET_PARTITIONS = (5, 6)

os.environ["FINN_BUILD_DIR"] = os.path.join(ENET_DIR, "finn_build_tmp")
os.makedirs(SCRATCH_DIR, exist_ok=True)

flat_ckpt = os.path.join(PREAMBLE_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
flat_model = ModelWrapper(flat_ckpt)

# Re-split into a SCRATCH dir only -- partitions 0-4/7's real, already-built
# checkpoints under REAL_OUTPUT_DIR are never written by this call.
cfg_scratch = dataclasses.replace(buildmod.base.cfg_stitched_ip_partitioned_8way, output_dir=SCRATCH_DIR)
parent_model = step_create_dataflow_partition_multi(flat_model, cfg_scratch)
sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"
print("Regenerated fresh (unbuilt) partitions in scratch dir:", SCRATCH_DIR)
for i, n in enumerate(sdp_nodes):
    print(f"  [{i}] {n.name} -> {getCustomOp(n).get_nodeattr('model')}")

with open(buildmod.FOLDING_BLOCK_FILE) as f:
    per_layer = json.load(f)["per_layer"]

logical_by_partition = buildmod.load_all_partition_logical_names(PREAMBLE_DIR)

backup_dir = os.path.join(REAL_OUTPUT_DIR, f"archive_pre_{datetime.now().strftime('%Y%m%d')}_p5p6_folding_fix")
os.makedirs(backup_dir, exist_ok=True)

for i in TARGET_PARTITIONS:
    sdp_node = sdp_nodes[i]
    scratch_partition_fn = getCustomOp(sdp_node).get_nodeattr("model")
    conv_names, pool_names = logical_by_partition[i]

    folding_config, n_unmatched = buildmod.build_partition_folding_config(
        PREAMBLE_DIR, i, sdp_node.name, scratch_partition_fn, conv_names, pool_names, per_layer, SCRATCH_DIR,
    )
    print(f"[partition {i}] corrected folding config derived, {n_unmatched} unmatched")

    # Reload a fresh copy of the raw scratch partition and apply the same
    # sequence for real -- build_partition_folding_config's own internal
    # specialize/fold pass above was throwaway (used only to derive the
    # dict), mirrors _build_and_synth_one_partition's real order up to (not
    # including) step_hw_codegen.
    kernel_model = ModelWrapper(scratch_partition_fn)
    kernel_model = step_specialize_layers(kernel_model, cfg_scratch)
    kernel_model = kernel_model.transform(GiveUniqueNodeNames())
    kernel_model = kernel_model.transform(GiveReadableTensorNames())
    kernel_model = step_target_fps_parallelization(kernel_model, cfg_scratch)

    folding_json_path = os.path.join(SCRATCH_DIR, f"hawq_folding_config_partition{i}_FIXED.json")
    with open(folding_json_path, "w") as f:
        json.dump(folding_config, f, indent=2)
    part_cfg = dataclasses.replace(cfg_scratch, folding_config_file=folding_json_path)
    kernel_model = step_apply_folding_config(kernel_model, part_cfg)

    # Verify the fix actually landed on every VVAU/SWU pair in this partition.
    for node in kernel_model.graph.node:
        if node.op_type in buildmod.VVAU_OP_TYPES:
            inst = getCustomOp(node)
            pe, simd = inst.get_nodeattr("PE"), inst.get_nodeattr("SIMD")
            _, swu_node = buildmod.find_preceding_swu_fmpad(kernel_model, node)
            swu_inst = getCustomOp(swu_node)
            print(f"  VERIFY [partition {i}] {node.name}: PE={pe} SIMD={simd} | "
                  f"SWU {swu_node.name}: SIMD={swu_inst.get_nodeattr('SIMD')} "
                  f"parallel_window={swu_inst.get_nodeattr('parallel_window')}")

    # Back up the real (crashed, post-synth-attempt) checkpoint, then
    # overwrite ONLY this partition's real files.
    real_partition_fn = os.path.join(
        REAL_OUTPUT_DIR, "intermediate_models", "supported_op_partitions", f"partition_{i}.onnx"
    )
    backup_fn = os.path.join(backup_dir, f"partition_{i}.onnx")
    shutil.copy2(real_partition_fn, backup_fn)
    print(f"[partition {i}] backed up crashed checkpoint -> {backup_fn}")

    kernel_model.save(real_partition_fn)
    print(f"[partition {i}] saved CORRECTED pre-fifo-autosizing graph -> {real_partition_fn}")

    real_folding_json = os.path.join(REAL_OUTPUT_DIR, f"hawq_folding_config_partition{i}.json")
    if os.path.exists(real_folding_json):
        shutil.copy2(real_folding_json, os.path.join(backup_dir, f"hawq_folding_config_partition{i}.json"))
    with open(real_folding_json, "w") as f:
        json.dump(folding_config, f, indent=2)
    print(f"[partition {i}] refreshed folding config json -> {real_folding_json}")

print("\nDone. Partitions 5 and 6 in", REAL_OUTPUT_DIR, "now hold the corrected pre-FIFO-autosizing graph.")
print("Partitions 0, 1, 2, 3, 4 and 7 were never touched.")
print("NEXT STEP (not run by this script): resume the build for partitions 5/6 only from")
print("step_minimize_bit_width_standalone_thresh_aware onward (hw_codegen -> hw_ipgen -> ")
print("set_fifo_depths -> SplitLargeFIFOs -> stitched IP -> OOC synth).")
