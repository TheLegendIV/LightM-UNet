"""Compare saturated- vs paced-source FIFO autosizing for the same partitions (FINN container).

Usage: python3 compare_fifo_sources.py <sat supported_op_partitions dir> <paced supported_op_partitions dir> <out.json> <ids...>

Checks that both checkpoints have the same folding (PE/SIMD per compute node) so that only the source differs, then
compares FIFO depths per edge (an edge missing on one side was a shallow FIFO removed by RemoveShallowFIFOs).
"""
import json
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from dump_fifo_depths import dump_partition


def folding(path):
    m = ModelWrapper(path)
    out = {}
    for n in m.graph.node:
        if n.op_type.startswith("StreamingFIFO"):
            continue
        op = getCustomOp(n)
        out[n.name] = tuple(
            int(op.get_nodeattr(a)) for a in ("PE", "SIMD") if a in op.get_nodeattr_types()
        )
    return out


def top(rows, n=6):
    return sorted(rows, key=lambda r: -r["kbit"])[:n]


if __name__ == "__main__":
    sat_dir, pac_dir, out = sys.argv[1:4]
    ids = [int(x) for x in sys.argv[4:]]
    report = {}
    for i in ids:
        fs, fp = f"{sat_dir}/partition_{i}_postfifo_autosize.onnx", f"{pac_dir}/partition_{i}_postfifo_autosize.onnx"
        fold_s, fold_p = folding(fs), folding(fp)
        diff_fold = [k for k in fold_s if fold_s[k] != fold_p.get(k)]
        sat = {f["edge"]: f for f in dump_partition(fs)}
        pac = {f["edge"]: f for f in dump_partition(fp)}
        rows = []
        for e in sorted(set(sat) | set(pac)):
            s, p = sat.get(e), pac.get(e)
            rows.append({
                "edge": e,
                "sat_depth": s["real_depth"] if s else 0,
                "paced_depth": p["real_depth"] if p else 0,
                "width_bits": (s or p)["width_bits"],
                "sat_kbit": s["kbit"] if s else 0.0,
                "paced_kbit": p["kbit"] if p else 0.0,
            })
        tot_s = sum(r["sat_kbit"] for r in rows) / 1024
        tot_p = sum(r["paced_kbit"] for r in rows) / 1024
        big = [r for r in rows if max(r["sat_depth"], r["paced_depth"]) >= 4096]
        report[str(i)] = {"folding_mismatch_nodes": diff_fold, "sat_mbit": tot_s, "paced_mbit": tot_p,
                          "n_fifos_sat": len(sat), "n_fifos_paced": len(pac), "edges": rows}
        print(f"\n=== partition {i}: folding mismatches {len(diff_fold)} | FIFOs sat {len(sat)} paced {len(pac)} | "
              f"total sat {tot_s:.2f} Mbit -> paced {tot_p:.2f} Mbit ({100 * (tot_p / tot_s - 1):+.1f}%)")
        print(f"{'edge':70s} {'width':>5s} {'sat':>9s} {'paced':>9s}")
        for r in sorted(big, key=lambda r: -max(r["sat_kbit"], r["paced_kbit"])):
            print(f"{r['edge'][:70]:70s} {r['width_bits']:5d} {r['sat_depth']:9d} {r['paced_depth']:9d}")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
