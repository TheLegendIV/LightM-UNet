"""Preamble (no-Vivado) for the 12_dense_relu_warmstart150ep alpha=0.25
trained build, IDENTICAL to
finn_hawq_preamble_12_dense_relu_warmstart150ep_alpha025_trained.py except
it swaps in step_enet_convert_to_hw_rtl_mvau (finn_enet_convert_to_hw_rtl_mvau.py)
in place of the standard step_enet_convert_to_hw, forcing every MVAU/VVAU
node to noActivation=1 (standalone Thresholding, not fused) -- a
prerequisite for RTL specialization by step_specialize_layers. Purely
structural check (no Vivado) -- run this first and verify noActivation
lands as 1 everywhere before committing to a full 8-way OOC synth build.

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_hawq_preamble_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau.py
"""
import os
import sys
import dataclasses
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
from finn_enet_convert_to_hw_rtl_mvau import step_enet_convert_to_hw_rtl_mvau  # noqa: E402
import finn.builder.build_dataflow as build  # noqa: E402

import finn_stage_partition  # noqa: E402

_orig_find_stage_boundaries = finn_stage_partition.find_stage_boundaries


def _find_stage_boundaries_relaxed(model):
    """Same as finn_stage_partition.find_stage_boundaries, minus the
    FMPadding_Pixel count assert -- that assert is documented in the
    original as "no longer used to derive up4_start/up5_start", a
    fail-fast sanity check only. Forcing standalone thresholds
    (step_enet_convert_to_hw_rtl_mvau) changes how many FMPadding_Pixel
    nodes survive (5 vs the expected 3, observed 2026-09-16) since fewer
    activations get fused away before InferConvInpGen runs -- harmless to
    the actual boundary computation, which only uses StreamingMaxPool/
    UpsampleNearestNeighbour node counts."""
    fmpad = model.get_nodes_by_op_type("FMPadding_Pixel")
    if len(fmpad) != 3:
        print(f"[find_stage_boundaries_relaxed] found {len(fmpad)} FMPadding_Pixel "
              "nodes (expected 3) -- ignoring, not used to derive boundaries.")
    maxpools = model.get_nodes_by_op_type("StreamingMaxPool")
    assert len(maxpools) in (2, 3), (
        "Expected 2 (down1, down2) or 3 (+ initial's own pool branch) "
        "StreamingMaxPool nodes, found %d." % len(maxpools)
    )
    upsample = [n for n in model.graph.node if n.op_type.startswith("UpsampleNearestNeighbour")]
    assert len(upsample) == 2, (
        "Expected exactly 2 UpsampleNearestNeighbour_* nodes, found %d." % len(upsample)
    )
    node_index = lambda n: list(model.graph.node).index(n)  # noqa: E731
    maxpools = sorted(maxpools, key=node_index)
    upsample = sorted(upsample, key=node_index)
    down1_start = node_index(maxpools[-2])
    down2_start = node_index(maxpools[-1])
    up4_start = node_index(upsample[0])
    up5_start = node_index(upsample[1])
    boundaries = [down1_start, down2_start, up4_start, up5_start]
    assert boundaries == sorted(boundaries), (
        "Detected stage boundaries are not in ascending topological order (%s)" % boundaries
    )
    return boundaries


finn_stage_partition.find_stage_boundaries = _find_stage_boundaries_relaxed

MODEL_NAME = "quantEnet_12_dense_relu_warmstart150ep_alpha025_trained_int8"
MODEL_FILE = os.path.join(base.ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(
    base.ENET_DIR, "finn_deployment_outputs",
    f"12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_preamble_{timestamp}",
)

idx_convert = base.enet_ip_partitioned_8way_steps.index(base.step_enet_convert_to_hw)
idx_partition = base.enet_ip_partitioned_8way_steps.index(base.assign_stage_partition_ids_8way)
steps = list(base.enet_ip_partitioned_8way_steps[: idx_partition + 1])
steps[idx_convert] = step_enet_convert_to_hw_rtl_mvau
print("Steps to run:", [s if isinstance(s, str) else s.__name__ for s in steps])

cfg = dataclasses.replace(
    base.cfg_stitched_ip_partitioned_8way,
    output_dir=OUTPUT_DIR,
    steps=steps,
    generate_outputs=[],
    save_intermediate_models=True,
)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("OUTPUT_DIR=", OUTPUT_DIR, flush=True)
    build.build_dataflow_cfg(MODEL_FILE, cfg)
    print("Done. Intermediate models in:", os.path.join(OUTPUT_DIR, "intermediate_models"))
    print("OUTPUT_DIR=", OUTPUT_DIR)

    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp

    ckpt = os.path.join(OUTPUT_DIR, "intermediate_models", "step_enet_convert_to_hw_rtl_mvau.onnx")
    m = ModelWrapper(ckpt)
    n_thresh = 0
    for node in m.graph.node:
        # Generic (pre-specialization) op_types -- step_specialize_layers
        # (MVAU/VVAU/Thresholding -> _hls/_rtl variants) hasn't run yet at
        # this checkpoint, it happens per-partition later.
        if node.op_type in ("MVAU", "VVAU"):
            inst = getCustomOp(node)
            wdt = inst.get_nodeattr("weightDataType")
            no_act = inst.get_nodeattr("noActivation")
            print(f"{node.name:30s} {node.op_type:10s} noActivation={no_act} weightDataType={wdt}")
        elif node.op_type == "Thresholding":
            n_thresh += 1
    print(f"standalone Thresholding nodes: {n_thresh}")
