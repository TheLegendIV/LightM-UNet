"""Post-solve reports for a finn_milp.py result. Agents: read finn_milp.md
("Outputs") before changing what these files contain.

  pruning_<stem>.json  -- pruning candidates: blocks holding zero-sensitivity
                          layers first, then every prunable block ranked by
                          measured sensitivity, each with what pruning it frees.
  final_output.onnx    -- the solved dataflow graph (every FINN node: convs, pools,
                          thresholds, stream nodes), op_type set to the real FINN
                          v0.10.1 custom-op name (MVAU_rtl, Thresholding_rtl, ...),
                          with folding/bits/cycles/II/resources as node attributes.
                          Open in Netron.

Both are written automatically by finn_milp.py for an Optimal solve. To
regenerate from an existing result:
    python MILP/milp_outputs.py --result <layer_bits_folding_*.json> --sensitivity-file <layer_sensitivity_*.json>
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import TensorProto, helper

OUT_ACT_SUFFIX = ".out_act"
DUP_SUFFIX = ".dup"
SKIP_QUANT_SUFFIX = ".skip_quant"
ZERO_SENSITIVITY_EPS = 1e-9


def _layer_op_type(op: str, variant: str) -> str:
    """Real FINN v0.10.1 op name for a per-layer (conv/pool) node -- extra
    nodes already carry theirs via finn_milp.EXTRA_OP_LABEL (op_types). MVAU
    is rtl iff its own chosen resource variant is the RTL one (finn_milp.py's
    VARIANT_RTL_DSP_NOACT1 vs VARIANT_HLS_LUT_NOACT0, the latter only
    reachable via --allow-lut-mult, off by default)."""
    if op == "MaxPool2d":
        return "StreamingMaxPool_hls"
    return "MVAU_rtl" if variant.startswith("rtl") else "MVAU_hls"


def _collect_nodes(result: dict) -> dict[str, dict]:
    """One flat record per hardware node (convs, pools, and every extra dataflow node)."""
    op_types = result["dataflow_graph"]["op_types"]
    nodes = {}
    for name, v in result["per_layer"].items():
        nodes[name] = {
            "op": op_types[name], "stage": v["stage"], "pe": v["pe"], "simd": v["simd"],
            "weight_bits": v["weight_bits"], "act_bits": v["act_bits"], "cycles": v["cycles"],
            "lut": v["lut_calibrated"], "bram18k": v["bram18k_calibrated"], "dsp": v["total_dsp"],
            "uram18": v.get("wm_uram18", 0) + v.get("swu_uram18", 0) + v.get("thr_uram18", 0),
            "thr_ram_style": v["thr_ram_style"], "variant": v["variant"],
            "mvu_cycles": v.get("mvu_cycles"), "swu_cycles": v.get("swu_cycles"), "thr_pe": v.get("thr_pe"),
        }
    for name, v in result.get("extra_nodes", {}).items():
        nodes[name] = {
            "op": op_types[name], "kind": v["kind"], "stage": v["stage"], "pe": v["pe"], "simd": v["simd"],
            "weight_bits": v["weight_bits"], "act_bits": v["act_bits"], "cycles": v["cycles"],
            "lut": v["lut_calibrated"], "bram18k": v["bram18k_calibrated"], "dsp": v["dsp"], "uram18": v["uram18"],
            "thr_ram_style": v["ram_style"], "bits_rule": v["bits_rule"],
        }
    return nodes


def _resources(nodes: list[dict]) -> dict:
    return {
        "lut_calibrated": sum(n["lut"] for n in nodes), "bram18k_calibrated": sum(n["bram18k"] for n in nodes),
        "dsp": sum(n["dsp"] for n in nodes), "cycles": sum(n["cycles"] for n in nodes),
    }


def build_pruning_report(result: dict, sensitivity: dict, zero_sensitivity_layers: list[str]) -> dict:
    """A block is prunable only when it is a residual block with an identity
    skip (its skip_quant reads the previous block's out_act) -- the only case
    ENet.apply_block_pruning can replace with nn.Identity soundly (in == out
    channels). Downsampling/upsampling blocks change shape and never are."""
    edges = result["dataflow_graph"]["edges"]
    nodes = _collect_nodes(result)
    by_block: dict[str, list[str]] = defaultdict(list)
    for name, n in nodes.items():
        by_block[n["stage"]].append(name)

    def identity_skip(block: str) -> bool:
        preds = [edges[p][0] if p.endswith(DUP_SUFFIX) else p for p in edges.get(block + SKIP_QUANT_SUFFIX, [])]
        return bool(preds) and all(p.endswith(OUT_ACT_SUFFIX) for p in preds)

    zero = set(zero_sensitivity_layers)
    diag = result["_diagnostics"]
    entries = {}
    for block, names in by_block.items():
        weighted = [n for n in names if n in sensitivity]
        convs = [n for n in names if nodes[n]["op"] in ("Conv2d", "ConvTranspose2d") and "kind" not in nodes[n]]
        frees = _resources([nodes[n] for n in names])
        entries[block] = {
            "block": block,
            "prunable": identity_skip(block),
            "zero_sensitivity_layers": sorted(zero & set(names)),
            "all_measured_layers_zero": bool(weighted) and all(n in zero for n in weighted),
            "block_sensitivity": sum(abs(sensitivity[n]["trace_w"]) + abs(sensitivity[n]["trace_a"]) for n in weighted),
            "min_conv_out_channels": min((result["dataflow_graph"]["shapes"][n][0] for n in convs), default=None),
            "frees": frees,
            "frees_pct_of_budget": {
                "lut": 100 * frees["lut_calibrated"] / diag["xczu7ev_lut_budget"],
                "bram18k": 100 * frees["bram18k_calibrated"] / diag["xczu7ev_bram18k_budget"],
                "dsp": 100 * frees["dsp"] / diag["xczu7ev_dsp_budget"],
                "cycles": 100 * frees["cycles"] / diag["total_cycles"],
            },
            "env": f"ENET_PRUNED_BLOCKS={block}" if identity_skip(block) else None,
        }

    candidates = sorted(
        (e for e in entries.values() if e["zero_sensitivity_layers"]),
        key=lambda e: (not e["prunable"], not e["all_measured_layers_zero"], e["block_sensitivity"]),
    )
    for e in candidates:
        if not e["prunable"]:
            e["reason"] = "not an identity-skip residual block -- apply_block_pruning would change the tensor shape"
        elif e["all_measured_layers_zero"]:
            e["reason"] = "every measured layer has zero HAWQ trace -- the block contributes nothing measurable"
        else:
            e["reason"] = "some layers have zero trace -- inspect before pruning the whole block"
    ranked = sorted((e for e in entries.values() if e["prunable"]), key=lambda e: e["block_sensitivity"])
    return {
        "note": "block_sensitivity = sum over the block's measured layers of |trace_w| + |trace_a| "
                "(layer_sensitivity.py). frees = this solve's resources/cycles for every hardware node in the "
                "block, incl. its residual-join thresholds. Pruning needs retraining/QAT to confirm accuracy.",
        "zero_sensitivity_layers": sorted(zero),
        "candidates": candidates,
        "ranked_prunable_blocks": ranked,
    }


def _topological_order(names: set[str], edges: dict[str, list[str]]) -> list[str]:
    indegree = {n: sum(p in names for p in edges.get(n, [])) for n in names}
    successors: dict[str, list[str]] = defaultdict(list)
    for n in names:
        for p in edges.get(n, []):
            if p in names:
                successors[p].append(n)
    ready = sorted(n for n, d in indegree.items() if d == 0)
    order = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for s in successors[n]:
            indegree[s] -= 1
            if indegree[s] == 0:
                ready.append(s)
    if len(order) != len(names):
        raise ValueError("dataflow graph has a cycle")
    return order


def export_dataflow_onnx(result: dict, path: Path) -> None:
    graph_info = result["dataflow_graph"]
    edges, shapes = graph_info["edges"], graph_info["shapes"]
    nodes = _collect_nodes(result)
    order = _topological_order(set(nodes), edges)
    max_cycles = max(n["cycles"] for n in nodes.values())
    consumed = {p for n in nodes for p in edges.get(n, []) if p in nodes}

    onnx_nodes, value_infos, outputs = [], [], []
    for name in order:
        n = nodes[name]
        channels, height, width = shapes[name]
        inputs = [f"{p}:out" for p in edges.get(name, []) if p in nodes] or ["global_in"]
        attrs = {
            "stage": n["stage"], "shape_CHW": f"{channels}x{height}x{width}", "pe": n["pe"], "simd": n["simd"],
            "cycles": int(n["cycles"]),
            "ii_cycles_per_pixel": n["cycles"] / (height * width),
            "rate_elems_per_cycle": channels * height * width / n["cycles"],
            # NOT the same as ii_cycles_per_pixel: this is channel-normalized (cycles per
            # individual scalar element), exactly finn_milp.py's chain-coherence rate_expr --
            # so nodes with different channel counts stay comparable, and a node's own value
            # here directly checked against ratio*(its slowest descendant's value) is exactly
            # what --optimize-downstream-rate enforced during the solve.
            "chain_rate_cycles_per_elem": n["cycles"] / (channels * height * width),
            "pct_of_slowest_node": 100 * n["cycles"] / max_cycles, "is_slowest_node": int(n["cycles"] == max_cycles),
            "lut": float(n["lut"]), "bram18k": float(n["bram18k"]), "dsp": int(n["dsp"]), "uram18": float(n["uram18"]),
            "thr_ram_style": n["thr_ram_style"],
        }
        for key in ("act_bits", "weight_bits", "variant", "kind", "bits_rule", "mvu_cycles", "swu_cycles", "thr_pe"):
            if n.get(key) is not None:
                attrs[key] = n[key]
        op_type = n["op"] if "kind" in n else _layer_op_type(n["op"], n["variant"])
        onnx_nodes.append(helper.make_node(
            op_type, inputs, [f"{name}:out"], name=name, domain="finn_milp", **attrs,
        ))
        info = helper.make_tensor_value_info(f"{name}:out", TensorProto.FLOAT, [1, channels, height, width])
        (value_infos if name in consumed else outputs).append(info)

    diag = result["_diagnostics"]
    summary = (
        f"status={result['status']} alpha={result['alpha']} | LUT {diag['lut_pct_of_budget']:.1f}% "
        f"BRAM {diag['bram_pct_of_budget']:.1f}% DSP {diag['dsp_pct_of_budget']:.1f}% | "
        f"total_cycles={diag['total_cycles']:.0f} | slowest node cycles={max_cycles:.0f}"
    )
    graph = helper.make_graph(
        onnx_nodes, "finn_milp_dataflow",
        [helper.make_tensor_value_info("global_in", TensorProto.FLOAT, [1, *graph_info["input"]])],
        outputs, value_info=value_infos, doc_string=summary,
    )
    model = helper.make_model(
        graph, producer_name="finn_milp",
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("finn_milp", 1)],
    )
    onnx.save(model, str(path))


def write_outputs(result: dict, out_file: Path, sensitivity: dict, zero_sensitivity_layers: list[str]) -> list[Path]:
    stem = out_file.stem
    pruning_path = out_file.parent / f"pruning_{stem}.json"
    onnx_path = out_file.parent / "final_output.onnx"
    pruning_path.write_text(json.dumps(build_pruning_report(result, sensitivity, zero_sensitivity_layers), indent=2))
    export_dataflow_onnx(result, onnx_path)
    return [pruning_path, onnx_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", type=Path, required=True, help="An Optimal layer_bits_folding_*.json.")
    parser.add_argument("--sensitivity-file", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    if result["status"] != "Optimal" or "dataflow_graph" not in result:
        raise ValueError(f"{args.result}: needs an Optimal result written by a finn_milp.py that stores dataflow_graph.")
    sensitivity = json.loads(args.sensitivity_file.read_text())
    zero = sensitivity.pop("_zero_sensitivity_layers", None)
    if zero is None:
        zero = [
            n for n, e in sensitivity.items()
            if abs(e["trace_w"]) < ZERO_SENSITIVITY_EPS and abs(e["trace_a"]) < ZERO_SENSITIVITY_EPS
        ]
    for path in write_outputs(result, args.result, sensitivity, zero):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
