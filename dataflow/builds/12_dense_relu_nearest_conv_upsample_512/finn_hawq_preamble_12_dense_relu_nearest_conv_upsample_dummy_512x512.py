"""Preamble (no-Vivado) for the 12_dense_relu_nearest_conv_upsample DUMMY
build, 512x512 input -- run this FIRST (before the real ft15ep QAT
checkpoint lands) purely to discover structural/graph-conversion errors
early, per user instruction. IDENTICAL in structure/steps to
finn_hawq_preamble_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_512x512.py
(same shared step_enet_convert_to_hw_rtl_mvau conversion, same relaxed
stage-boundary detection, same dangling-node check) -- only MODEL_NAME/
MODEL_FILE and OUTPUT_DIR differ. Once the real ft15ep checkpoint lands,
copy this file to a "_trained" sibling pointing at
finn_export_12_dense_relu_nearest_conv_upsample_trained.py's own
--input-hw 512 512 export instead.

Prerequisite (run inside the pytorch training container first):
    docker exec lightm_pytorch python /workspace/LightM-UNet/hardware/finn_export_12_dense_relu_nearest_conv_upsample_dummy.py --input-hw 512 512
    docker cp hardware/outputs/finn_exports/quantEnet_12_dense_relu_nearest_conv_upsample_dummy_int8_512x512.onnx \\
        <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> bash -c \\
        "cd /home/thelegendiv/finn/notebooks/enet && python3 finn_hawq_preamble_12_dense_relu_nearest_conv_upsample_dummy_512x512.py"
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
    """Verbatim copy of the same-name helper in
    finn_hawq_preamble_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_512x512.py
    -- see that file's docstring for the full rationale. Still valid here:
    decoder_type="nearest_conv_upsample"'s main_up is ALSO a bare
    nn.Upsample(mode="nearest") (see LayerQuantEnetFINN.py's
    FINNUpsamplingBottleneck), so up4/up5 still each produce exactly one
    UpsampleNearestNeighbour_* node the same way the old decoder's frozen
    substitute did."""
    fmpad = model.get_nodes_by_op_type("FMPadding_Pixel")
    if len(fmpad) != 3:
        print(f"[find_stage_boundaries_relaxed] found {len(fmpad)} FMPadding_Pixel "
              "nodes (expected 3) -- ignoring, not used to derive boundaries.")
    maxpools = model.get_nodes_by_op_type("StreamingMaxPool")
    assert len(maxpools) in (2, 3), (
        "Expected 2 (down1, down2) or 3 (+ initial's own pool branch) "
        "StreamingMaxPool nodes, found %d." % len(maxpools)
    )
    node_index = lambda n: list(model.graph.node).index(n)  # noqa: E731
    maxpools = sorted(maxpools, key=node_index)
    down1_start = node_index(maxpools[-2])
    down2_start = node_index(maxpools[-1])

    upsample = [n for n in model.graph.node if n.op_type.startswith("UpsampleNearestNeighbour")]
    if len(upsample) == 2:
        upsample = sorted(upsample, key=node_index)
        up4_start = node_index(upsample[0])
        up5_start = node_index(upsample[1])
    else:
        print(f"[find_stage_boundaries_relaxed] found {len(upsample)} "
              "UpsampleNearestNeighbour_* nodes (expected 2) -- falling back "
              "to FMPadding_Pixel-pair detection for up4_start/up5_start.")
        fmpad_idx = sorted(node_index(n) for n in fmpad)
        groups = []
        for idx in fmpad_idx:
            if groups and idx - groups[-1][-1] <= 5:
                groups[-1].append(idx)
            else:
                groups.append([idx])
        pair_groups = [g for g in groups if len(g) >= 2]
        assert len(pair_groups) == 2, (
            "Expected exactly 2 FMPadding_Pixel pairs (up4.up, up5.up) as a "
            "fallback for missing Upsample nodes, found %d qualifying groups "
            "(all groups: %s)." % (len(pair_groups), groups)
        )
        up4_start, up5_start = pair_groups[0][0], pair_groups[1][0]

    boundaries = [down1_start, down2_start, up4_start, up5_start]
    assert boundaries == sorted(boundaries), (
        "Detected stage boundaries are not in ascending topological order (%s)" % boundaries
    )
    return boundaries


finn_stage_partition.find_stage_boundaries = _find_stage_boundaries_relaxed


def _check_dangling_nodes(model):
    """Traversal-based dangling-node checker -- verbatim copy, see sibling
    preamble script for the full rationale."""
    nodes = list(model.graph.node)
    graph_inputs = [i.name for i in model.graph.input]
    graph_outputs = set(o.name for o in model.graph.output)

    fwd_tensors, fwd_nodes = set(graph_inputs), set()
    queue = list(graph_inputs)
    while queue:
        for c in (model.find_consumers(queue.pop()) or []):
            if c.name not in fwd_nodes:
                fwd_nodes.add(c.name)
                for o in c.output:
                    if o not in fwd_tensors:
                        fwd_tensors.add(o)
                        queue.append(o)

    bwd_tensors, bwd_nodes = set(graph_outputs), set()
    queue = list(graph_outputs)
    while queue:
        t = queue.pop()
        p = model.find_producer(t)
        if p is None or p.name in bwd_nodes:
            continue
        bwd_nodes.add(p.name)
        for i in p.input:
            if model.get_initializer(i) is None and i not in bwd_tensors:
                bwd_tensors.add(i)
                queue.append(i)

    report = {"total_nodes": len(nodes), "dangling_nodes": []}
    for n in nodes:
        reachable_from_input = n.name in fwd_nodes
        reaches_output = n.name in bwd_nodes
        if not (reachable_from_input and reaches_output):
            report["dangling_nodes"].append({
                "name": n.name, "op_type": n.op_type,
                "reachable_from_input": reachable_from_input,
                "reaches_output": reaches_output,
            })
    report["n_dangling"] = len(report["dangling_nodes"])
    return report


MODEL_NAME = "quantEnet_12_dense_relu_nearest_conv_upsample_dummy_int8_512x512"
MODEL_FILE = os.path.join(base.ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(
    base.ENET_DIR, "finn_deployment_outputs",
    f"12_dense_relu_nearest_conv_upsample_dummy_rtl_mvau_512x512_preamble_{timestamp}",
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
        if node.op_type in ("MVAU", "VVAU"):
            inst = getCustomOp(node)
            wdt = inst.get_nodeattr("weightDataType")
            no_act = inst.get_nodeattr("noActivation")
            print(f"{node.name:30s} {node.op_type:10s} noActivation={no_act} weightDataType={wdt}")
        elif node.op_type == "Thresholding":
            n_thresh += 1
    print(f"standalone Thresholding nodes: {n_thresh}")

    import json

    final_ckpt = os.path.join(OUTPUT_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
    final_model = ModelWrapper(final_ckpt)
    dangling_report = _check_dangling_nodes(final_model)
    report_path = os.path.join(OUTPUT_DIR, "intermediate_models", "dangling_node_report.json")
    with open(report_path, "w") as f:
        json.dump(dangling_report, f, indent=2)
    print(f"dangling-node check: {dangling_report['n_dangling']} / {dangling_report['total_nodes']} "
          f"nodes dangling -- report written to {report_path}")
    if dangling_report["n_dangling"]:
        for d in dangling_report["dangling_nodes"]:
            print("  DANGLING:", d)
