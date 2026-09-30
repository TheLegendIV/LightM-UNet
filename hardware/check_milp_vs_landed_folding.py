"""Check that a MILP folding JSON landed unchanged in FINN's built graph.

Compares MILP/finn_milp.py's layer_bits_folding_<tag>.json ("per_layer") against
a FINN partition checkpoint (e.g. post_fifo_autosize_checkpoints/
partition*_<tag>_milpfold_prefifo_autosize.onnx). Per MVAU node (in graph order):
  - PE, SIMD, cycles_estimate  vs MILP pe, simd, mvu_cycles
  - preceding SWU (if any) SIMD vs MILP simd_swu
  - following standalone Thresholding PE vs MILP thr_pe (when the MILP sets it)
Exits 1 on any mismatch. The MILP layer window is found automatically: the
partition is a contiguous run of the MILP's MVAU-type layers, so the offset
with the most MVAU matches is used.

Needs only `onnx` (no FINN). Example, from the repo root:
    python hardware/check_milp_vs_landed_folding.py \\
        MILP/artifacts/<dir>/<tag>/layer_bits_folding_<tag>.json \\
        hardware/builds/<build>/post_fifo_autosize_checkpoints/partition2_<tag>_milpfold_prefifo_autosize.onnx
Run in `docker exec lightmunet_dev python3 ...` if the host python has no onnx.
"""
from __future__ import annotations

import argparse
import json
import sys

import onnx

SKIP_OPS = ("StreamingDataWidthConverter", "StreamingFIFO")


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
            "thr": thr.name if thr else None, "thr_pe": _attrs(thr).get("PE") if thr else None,
        })
    return records


def milp_mvau_layers(per_layer: dict) -> list[tuple[str, dict]]:
    return [(k, e) for k, e in per_layer.items() if e.get("mvu_cycles") is not None]


def find_offset(layers: list, landed: list[dict]) -> int:
    def score(off):
        return sum((e["pe"], e["simd"], e["mvu_cycles"]) == (r["pe"], r["simd"], r["cycles"])
                   for (_, e), r in zip(layers[off:], landed))
    return max(range(len(layers) - len(landed) + 1), key=score)


def check(milp_json: str, onnx_path: str) -> list[str]:
    per_layer = json.load(open(milp_json))["per_layer"]
    landed = landed_mvaus(onnx.load(onnx_path))
    layers = milp_mvau_layers(per_layer)
    if len(landed) > len(layers):
        return [f"landed graph has {len(landed)} MVAUs but MILP only has {len(layers)} MVAU-type layers"]
    off = find_offset(layers, landed)
    print(f"{len(landed)} landed MVAUs matched against MILP layers starting at {layers[off][0]!r}")
    problems = []
    for (name, e), r in zip(layers[off:], landed):
        want = (e["pe"], e["simd"], e["mvu_cycles"])
        got = (r["pe"], r["simd"], r["cycles"])
        if want != got:
            problems.append(f"{name} -> {r['node']}: (PE,SIMD,cycles) MILP {want} != landed {got}")
        if r["swu"] is not None and e.get("simd_swu") is not None and r["swu_simd"] != e["simd_swu"]:
            problems.append(f"{name} -> {r['swu']}: SWU SIMD MILP {e['simd_swu']} != landed {r['swu_simd']} "
                            f"(landed SWU cycles {r['swu_cycles']}, MILP swu_cycles {e.get('swu_cycles')})")
        if r["thr"] is not None and e.get("thr_pe") is not None and r["thr_pe"] != e["thr_pe"]:
            problems.append(f"{name} -> {r['thr']}: Thresholding PE MILP {e['thr_pe']} != landed {r['thr_pe']}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("milp_json", help="MILP layer_bits_folding_<tag>.json")
    ap.add_argument("onnx_path", help="FINN partition .onnx (post specialize/folding)")
    args = ap.parse_args()
    problems = check(args.milp_json, args.onnx_path)
    for p in problems:
        print("MISMATCH", p)
    print(f"{len(problems)} mismatch(es)" if problems else "OK: landed folding matches MILP")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
