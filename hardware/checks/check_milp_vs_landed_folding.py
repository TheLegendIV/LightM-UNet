"""Check that a MILP folding JSON landed unchanged in FINN's built graph.

Compares MILP/finn_milp.py's layer_bits_folding_<tag>.json ("per_layer") against
a FINN partition checkpoint (e.g. post_fifo_autosize_checkpoints/
partition*_<tag>_milpfold_prefifo_autosize.onnx). Per MVAU node (in graph order):
  - PE, SIMD, cycles_estimate  vs MILP pe, simd, mvu_cycles
  - preceding SWU (if any) SIMD vs MILP simd_swu, parallel_window, and cycles vs MILP swu_cycles
  - following standalone Thresholding PE vs MILP thr_pe (when the MILP sets it)
Exits 1 on any mismatch. The MILP layer window is found automatically: the
partition is a contiguous run of the MILP's MVAU-type layers, so the offset
with the most MVAU matches is used.

Needs only `onnx` (no FINN). Example, from the repo root:
    python hardware/checks/check_milp_vs_landed_folding.py \\
        MILP/artifacts/<dir>/<tag>/layer_bits_folding_<tag>.json \\
        hardware/builds/<build>/post_fifo_autosize_checkpoints/partition2_<tag>_milpfold_prefifo_autosize.onnx \\
        --conv-order hardware/outputs/<model>_conv_order.json
Run in `docker exec lightmunet_dev python3 ...` if the host python has no onnx.

The matching between MILP per_layer keys (file order) and landed MVAU nodes
(real graph order) is positional by default, which is WRONG whenever a block
forks into sibling branches of different depth before the next weight-bearing
op (confirmed for FINNUpsamplingBottleneck's skip_resize_conv/reduce pair --
see finn_gotchas repo memory). Pass --conv-order to validate/self-correct
this against each landed node's real weight-initializer element count (the
same fix applied in finn_s12_build_steps.load_partition_logical_names);
without it, such swaps print as false MISMATCHes even though the landed fold
is actually correct.
"""
from __future__ import annotations

import argparse
import json
import sys

import onnx

SKIP_OPS = ("StreamingDataWidthConverter", "StreamingFIFO")
REALIGN_WINDOW = 12  # how far ahead to search for a weight-count match on mismatch


def _shape_count(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def load_conv_order_weight_counts(conv_order_file: str) -> dict[str, int]:
    entries = json.load(open(conv_order_file))
    return {e["logical_name"]: _shape_count(e["weight_shape"]) for e in entries if e.get("weight_shape")}


def _attrs(node) -> dict:
    return {a.name: a.i for a in node.attribute if a.type == onnx.AttributeProto.INT}


def _is_skip(node) -> bool:
    return node.op_type.startswith(SKIP_OPS)


def landed_mvaus(model) -> list[dict]:
    """One record per MVAU in graph order, with its SWU and Thresholding neighbours."""
    producer = {o: n for n in model.graph.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in model.graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    initializers = {init.name: init for init in model.graph.initializer}

    def weight_count(node):
        for inp in node.input:
            init = initializers.get(inp)
            if init is not None:
                return _shape_count(init.dims)
        return None

    def upstream(node):
        p = producer.get(node.input[0])
        while p is not None and _is_skip(p):
            p = producer.get(p.input[0])
        return p

    def downstream(node):
        c = consumers.get(node.output[0], [])
        while len(c) == 1 and _is_skip(c[0]):
            c = consumers.get(c[0].output[0], [])
        return c[0] if len(c) == 1 else None

    records = []
    for n in model.graph.node:
        if not n.op_type.startswith("MVAU"):
            continue
        swu = upstream(n)
        swu = swu if swu is not None and swu.op_type.startswith("ConvolutionInputGenerator") else None
        thr = downstream(n)
        thr = thr if thr is not None and thr.op_type.startswith("Thresholding") else None
        a = _attrs(n)
        records.append({
            "node": n.name, "pe": a.get("PE"), "simd": a.get("SIMD"), "cycles": a.get("cycles_estimate"),
            "swu": swu.name if swu else None, "swu_simd": _attrs(swu).get("SIMD") if swu else None,
            "swu_cycles": _attrs(swu).get("cycles_estimate") if swu else None,
            "swu_pw": _attrs(swu).get("parallel_window") if swu else None,
            "swu_ifm_ch": _attrs(swu).get("IFMChannels") if swu else None,
            "thr": thr.name if thr else None, "thr_pe": _attrs(thr).get("PE") if thr else None,
            "weight_count": weight_count(n),
        })
    return records


def milp_mvau_layers(per_layer: dict) -> list[tuple[str, dict]]:
    return [(k, e) for k, e in per_layer.items() if e.get("mvu_cycles") is not None]


def find_offset(layers: list, landed: list[dict]) -> int:
    def score(off):
        return sum((e["pe"], e["simd"], e["mvu_cycles"]) == (r["pe"], r["simd"], r["cycles"])
                   for (_, e), r in zip(layers[off:], landed))
    candidates = range(len(layers) - len(landed) + 1)
    best = max(candidates, key=score)
    tied = [off for off in candidates if score(off) == score(best)]
    if len(tied) > 1:
        print(f"WARNING: find_offset is ambiguous -- {len(tied)} offsets tie on (PE,SIMD,mvu_cycles) match count "
              f"(starts: {[layers[o][0] for o in tied]}). Picking the first ({layers[tied[0]][0]!r}) -- pass "
              "--start-layer explicitly if this is wrong (e.g. symmetric-compute architectures alias different "
              "partitions onto identical PE/SIMD/cycles).")
    return best


def check(milp_json: str, onnx_path: str, start_layer: str | None = None,
          weight_counts: dict[str, int] | None = None) -> list[str]:
    per_layer = json.load(open(milp_json))["per_layer"]
    landed = landed_mvaus(onnx.load(onnx_path))
    layers = milp_mvau_layers(per_layer)
    if len(landed) > len(layers):
        return [f"landed graph has {len(landed)} MVAUs but MILP only has {len(layers)} MVAU-type layers"]
    if start_layer is not None:
        names = [name for name, _ in layers]
        if start_layer not in names:
            return [f"--start-layer {start_layer!r} not found among MILP MVAU-type layers: {names}"]
        off = names.index(start_layer)
    else:
        off = find_offset(layers, landed)
    print(f"{len(landed)} landed MVAUs matched against MILP layers starting at {layers[off][0]!r}")
    if weight_counts is None:
        print("WARNING: no --conv-order given -- matching MILP layers to landed nodes purely positionally. "
              "This is WRONG for forked blocks of different depth (e.g. FINNUpsamplingBottleneck's "
              "skip_resize_conv/reduce pair) -- pass --conv-order to shape-validate and self-correct.")
    # Mutable window: realign (swap) MILP entries forward/back to match each landed node's real
    # weight-element count, mirroring finn_s12_build_steps.load_partition_logical_names's fix for
    # the same conv_order.json-vs-real-graph node-order divergence.
    window = list(layers[off:off + len(landed)])
    problems = []
    for i, r in enumerate(landed):
        name, e = window[i]
        expected = weight_counts.get(name) if weight_counts else None
        if expected is not None and r["weight_count"] is not None and expected != r["weight_count"]:
            match_j = next(
                (j for j in range(i + 1, min(i + 1 + REALIGN_WINDOW, len(window)))
                 if weight_counts.get(window[j][0]) == r["weight_count"]),
                None,
            )
            if match_j is None:
                problems.append(f"{name} -> {r['node']}: weight count MILP {expected} != landed "
                                f"{r['weight_count']}, and no matching MILP layer found within the next "
                                f"{REALIGN_WINDOW} entries")
                continue
            print(f"[checker] REALIGN: MILP layer {name!r} doesn't match landed {r['node']} (weight count "
                  f"{r['weight_count']}) -- swapping in {window[match_j][0]!r} instead.")
            window[i], window[match_j] = window[match_j], window[i]
            name, e = window[i]
        want = (e["pe"], e["simd"], e["mvu_cycles"])
        got = (r["pe"], r["simd"], r["cycles"])
        if want != got:
            problems.append(f"{name} -> {r['node']}: (PE,SIMD,cycles) MILP {want} != landed {got}")
        if r["swu"] is not None and e.get("simd_swu") is not None and r["swu_simd"] != e["simd_swu"]:
            problems.append(f"{name} -> {r['swu']}: SWU SIMD MILP {e['simd_swu']} != landed {r['swu_simd']} "
                            f"(landed SWU cycles {r['swu_cycles']}, MILP swu_cycles {e.get('swu_cycles')})")
        if r["swu"] is not None and e.get("swu_cycles"):
            # MILP's SWU model is ported from FINN's get_exp_cycles: parallel_window iff MVAU SIMD > cin.
            # (finn_cost_model.conv_cost_pe_simd: parallel_window = MVAU SIMD > cin)
            want_pw = r["swu_ifm_ch"] is not None and e["simd"] > r["swu_ifm_ch"]
            if r["swu_pw"] is not None and bool(r["swu_pw"]) != want_pw:
                problems.append(f"{name} -> {r['swu']}: SWU parallel_window landed {r['swu_pw']} but MILP prices it "
                                f"for {'parallel' if want_pw else 'non-parallel'} window (MVAU SIMD {e['simd']}, "
                                f"IFMChannels {r['swu_ifm_ch']})")
            if abs(r["swu_cycles"] - e["swu_cycles"]) > 0.01 * e["swu_cycles"] + 4:
                problems.append(f"{name} -> {r['swu']}: SWU cycles MILP {e['swu_cycles']} != landed {r['swu_cycles']}")
        if r["thr"] is not None and e.get("thr_pe") is not None and r["thr_pe"] != e["thr_pe"]:
            problems.append(f"{name} -> {r['thr']}: Thresholding PE MILP {e['thr_pe']} != landed {r['thr_pe']}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("milp_json", help="MILP layer_bits_folding_<tag>.json")
    ap.add_argument("onnx_path", help="FINN partition .onnx (post specialize/folding)")
    ap.add_argument("--start-layer", default=None,
                     help="MILP per_layer key the landed graph's first MVAU corresponds to. Overrides the "
                          "auto offset-finder -- needed when two partitions have identical (PE,SIMD,mvu_cycles) "
                          "(e.g. symmetric-compute-per-stage architectures), which makes find_offset ambiguous.")
    ap.add_argument("--conv-order", default=None,
                     help="conv_order.json for this model (gives each MILP layer's real weight-element count) "
                          "-- shape-validates the MILP-layer-to-landed-node match and self-corrects forked-branch "
                          "ordering swaps (e.g. skip_resize_conv/reduce) instead of printing false MISMATCHes.")
    args = ap.parse_args()
    weight_counts = load_conv_order_weight_counts(args.conv_order) if args.conv_order else None
    problems = check(args.milp_json, args.onnx_path, args.start_layer, weight_counts)
    for p in problems:
        print("MISMATCH", p)
    print(f"{len(problems)} mismatch(es)" if problems else "OK: landed folding matches MILP")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
