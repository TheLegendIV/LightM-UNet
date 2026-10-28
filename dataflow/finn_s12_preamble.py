"""S12-dense preamble (no Vivado): streamline -> rtl-mvau convert_to_hw
(standalone Thresholding) -> 8-way stage partition tagging.

Run inside the FINN container (onnx copied flat into notebooks/enet/):
    docker exec -e HOME=/tmp/home_dir <c> bash -c 'cd /home/thelegendiv/finn/notebooks/enet && \\
        python3 finn_s12_preamble.py <model_name_without_.onnx> --tag <tag> \\
        [--conv-order <conv_order.json> --folding-json <layer_bits_folding_*.json>]'
Output: finn_deployment_outputs/<tag>_preamble_<timestamp>/
With --folding-json, finn_s12_build.py --fifo-check (bridge + folding/FIFO landing gates, no HLS) runs automatically afterwards;
--no-fifo-check limits it to --bridge-only, --no-bridge skips it. The FIFO check also asserts that every node the bridge's FIFO role map names
is still the same kind of node (op type and channel/MH/MW shape) after DWC/FIFO insertion, and fails the preamble if not.
The MILP's per-pool implementation (per_layer[*.pool].pool_impl) selects the pool route of the conversion step (swg_pool for initial.pool = InferPool, see
finn_infer_pool_initial.py; it replaces setting INITIAL_POOL_ROUTE by hand) and finn_s12_build.py checks the lowered partitions against it.
"""
import argparse
import dataclasses
import json
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("model_name", help="onnx basename in notebooks/enet/ (without .onnx)")
_parser.add_argument("--tag", required=True, help="output dir prefix, e.g. S12_dense_nearest_upsample_512_hwsweep_wm_dsr_off")
_parser.add_argument(
    "--no-argmax", dest="argmax", action="store_false",
    help="skip the per-pixel top-1 in-PL argmax (step_insert_argmax_output/LabelSelect) and keep "
         "the raw 5-channel logit output instead. Default on.",
)
_parser.add_argument(
    "--blocks", action="store_true",
    help="tag one partition per analytical BLOCK (initial, down1, regular1.0, ..., final: 29) instead of the 8-way stage split, and write "
         "intermediate_models/block_partitions.json (finn_s12_blocks.py). Needs --conv-order. Build the blocks with finn_s12_build.py --blocks.",
)
_parser.add_argument("--conv-order", help="<model>_conv_order.json (required with --blocks and --folding-json)")
_parser.add_argument("--folding-json", help="MILP layer_bits_folding_*.json; when given, finn_s12_build.py --fifo-check runs on the new preamble (needs --conv-order)")
_parser.add_argument("--target-fps", type=float, default=None, help="forwarded to the bridge")
_parser.add_argument("--mvau-wwidth-max", type=int, default=None, help="forwarded to the bridge")
_parser.add_argument("--no-fifo-check", action="store_true", help="run only the bridge (--bridge-only), without the folding/FIFO landing gates")
_parser.add_argument("--no-bridge", action="store_true", help="skip the bridge/FIFO check even when --folding-json is given")
_args = _parser.parse_args()
if _args.blocks and not _args.conv_order:
    _parser.error("--blocks requires --conv-order")
if _args.folding_json and not _args.conv_order:
    _parser.error("--folding-json requires --conv-order")
if _args.folding_json and _args.blocks:
    _parser.error("--folding-json bridge is not supported with --blocks")


def apply_pool_route_from_milp(folding_json):
    """The MILP's per-pool implementation (per_layer[<pool>].pool_impl) decides how the conversion step lowers MaxPool: swg_pool for `initial.pool` -> INITIAL_POOL_ROUTE=infer_pool
    (finn_infer_pool_initial.py). A choice this flow cannot lower (InferPool in a down block) or an INITIAL_POOL_ROUTE that contradicts the MILP fails here, before any FINN step runs.
    MILP JSONs without pool_impl (solves before 2026-10-16) leave INITIAL_POOL_ROUTE as set."""
    with open(folding_json) as f:
        impls = {n: v["pool_impl"] for n, v in json.load(f).get("per_layer", {}).items() if n.endswith(".pool") and "pool_impl" in v}
    if not impls:
        print("pool route: MILP json has no pool_impl -- INITIAL_POOL_ROUTE=%r" % os.environ.get("INITIAL_POOL_ROUTE", ""))
        return
    unsupported = sorted(n for n, i in impls.items() if i == "swg_pool" and n != "initial.pool")
    if unsupported:
        sys.exit(f"MILP chose InferPool (swg_pool) for {unsupported}: this flow lowers InferPool only for initial.pool. Re-solve with finn_milp.py --pool-impl-scope initial.")
    route = "infer_pool" if impls.get("initial.pool") == "swg_pool" else ""
    have = os.environ.get("INITIAL_POOL_ROUTE", "")
    if have and have != route:
        sys.exit(f"INITIAL_POOL_ROUTE={have!r} contradicts the MILP's pool_impl {impls}")
    if route:
        os.environ["INITIAL_POOL_ROUTE"] = route
    else:
        os.environ.pop("INITIAL_POOL_ROUTE", None)
    print(f"pool route from the MILP: {impls} -> INITIAL_POOL_ROUTE={route!r}")


if _args.folding_json:
    apply_pool_route_from_milp(_args.folding_json)

# finn_enet_ip_build_partitioned_8way reads sys.argv[1]/[2] at import time.
_real_argv, sys.argv = sys.argv, sys.argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

import finn.builder.build_dataflow as build  # noqa: E402
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from finn_enet_convert_to_hw_rtl_mvau import step_enet_convert_to_hw_rtl_mvau  # noqa: E402
from finn_s12_build_steps import (  # noqa: E402
    check_dangling_nodes, install_relaxed_stage_boundaries, step_insert_argmax_output,
)
from finn_compose_thresholds import step_compose_consecutive_thresholds  # noqa: E402

import finn_s12_blocks  # noqa: E402

install_relaxed_stage_boundaries()


def main():
    model_file = os.path.join(base.ENET_DIR, f"{_args.model_name}.onnx")
    output_dir = os.path.join(
        base.ENET_DIR, "finn_deployment_outputs",
        f"{_args.tag}_preamble_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    idx_convert = base.enet_ip_partitioned_8way_steps.index(base.step_enet_convert_to_hw)
    idx_partition = base.enet_ip_partitioned_8way_steps.index(base.assign_stage_partition_ids_8way)
    steps = list(base.enet_ip_partitioned_8way_steps[: idx_partition + 1])
    # collapse residual_add->out_act (and any skip_quant/pool_quant chain) into one MultiThreshold
    # BEFORE hw conversion, so the real graph matches the analytical model's single Thr_out node
    # (validated 2026-10-05: 168->141 MultiThreshold, all 27 AddStreams get exactly 1 threshold after).
    steps.insert(idx_convert, step_compose_consecutive_thresholds)
    steps[idx_convert + 1] = step_enet_convert_to_hw_rtl_mvau
    if _args.argmax:
        # before stage-partition assignment: the new LabelSelect node's topological position
        # (after `final`'s ChannelwiseOp) puts it in partition 7 via assign_stage_partition_ids_8way's
        # own index-based fallback, same as any other trailing node.
        steps.insert(idx_convert + 2, step_insert_argmax_output)
    if _args.blocks:
        finn_s12_blocks.CONV_ORDER = _args.conv_order
        steps[idx_partition] = finn_s12_blocks.assign_block_partition_ids        # same position as assign_stage_partition_ids_8way
        print("per-block partitioning: assign_block_partition_ids (conv order %s)" % _args.conv_order)
    print("argmax:", _args.argmax)
    print("Steps to run:", [s if isinstance(s, str) else s.__name__ for s in steps])
    cfg = dataclasses.replace(
        base.cfg_stitched_ip_partitioned_8way,
        output_dir=output_dir, steps=steps, generate_outputs=[], save_intermediate_models=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    print("MODEL_FILE=", model_file, flush=True)
    print("OUTPUT_DIR=", output_dir, flush=True)
    build.build_dataflow_cfg(model_file, cfg)

    ckpt_dir = os.path.join(output_dir, "intermediate_models")
    m = ModelWrapper(os.path.join(ckpt_dir, "step_enet_convert_to_hw_rtl_mvau.onnx"))
    n_thresh = 0
    for node in m.graph.node:
        if node.op_type in ("MVAU", "VVAU"):
            inst = getCustomOp(node)
            print(f"{node.name:30s} {node.op_type:10s} noActivation={inst.get_nodeattr('noActivation')} "
                  f"weightDataType={inst.get_nodeattr('weightDataType')}")
        elif node.op_type == "Thresholding":
            n_thresh += 1
    print(f"standalone Thresholding nodes: {n_thresh}")

    flat_name = "assign_block_partition_ids.onnx" if _args.blocks else "assign_stage_partition_ids_8way.onnx"
    report = check_dangling_nodes(ModelWrapper(os.path.join(ckpt_dir, flat_name)))
    with open(os.path.join(ckpt_dir, "dangling_node_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"dangling-node check: {report['n_dangling']} / {report['total_nodes']} nodes dangling")
    for d in report["dangling_nodes"]:
        print("  DANGLING:", d)
    print("OUTPUT_DIR=", output_dir)
    return output_dir


def run_bridge(preamble_dir):
    """Bridge + folding/FIFO landing gates (seconds, no HLS) against the fresh preamble."""
    cmd = [sys.executable, "-u", os.path.join(os.path.dirname(os.path.abspath(__file__)), "finn_s12_build.py"),
           preamble_dir, "--tag", _args.tag, "--conv-order", _args.conv_order,
           "--folding-json", _args.folding_json, "--partitions", "all",
           "--bridge-only" if _args.no_fifo_check else "--fifo-check"]
    if _args.target_fps is not None:
        cmd += ["--target-fps", str(_args.target_fps)]
    if _args.mvau_wwidth_max is not None:
        cmd += ["--mvau-wwidth-max", str(_args.mvau_wwidth_max)]
    print("BRIDGE:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=base.ENET_DIR).returncode
    print(f"BRIDGE EXIT={rc}", flush=True)
    return rc


if __name__ == "__main__":
    out = main()
    if _args.folding_json and not _args.no_bridge:
        sys.exit(run_bridge(out))
    elif not _args.folding_json and not _args.blocks:
        print("bridge skipped: no --folding-json given")
