"""Summarise the ratchet-ablation arms (run run_ablation.sh first).

    python3 MILP/artifacts/S12_dense_256_ratchet_ablation_v1/summarize_arms.py

Writes next to this file: arms_summary.csv (one row per arm), arms_to_build.txt (one representative arm per DISTINCT folding, in arm order: arms with an identical folding share one hardware build)
and arm_groups.json (fingerprint -> arms). Also checks, per arm: the ratchet holds on every compute edge, mvau-wwidth-max 72 holds, every layer is INT6.
"""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ["ratchet_1pct", "ratchet_25pct", "ratchet_100pct", "ratchet_200pct", "ratchet_off", "analytical_25pct", "analytical_25pct_finnfifo"]
# hardware-only variants: same folding as their base arm, but the folding json is written WITHOUT the FIFO lists, so the FINN bridge forces nothing and FINN's own rtlsim autosizer sets every FIFO depth
VARIANTS = {"analytical_25pct_finnfifo": "analytical_25pct"}
WWIDTH_MAX, BITS = 72, 6


def load(arm):
    if arm in VARIANTS:                                   # generate the variant's folding json from its base arm (same folding, FIFO lists stripped)
        base = load(VARIANTS[arm])
        if base is None:
            return None
        d = {k: v for k, v in base.items() if k not in ("intra_block_fifos", "inter_block_fifos")}
        d["_variant"] = dict(of=VARIANTS[arm], note="FIFO lists removed: FINN's rtlsim autosizer (largefifo_rtlsim) sets every FIFO depth; the folding is identical to the base arm")
        out = HERE / arm / f"layer_bits_folding_{arm}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(d, indent=1))
        return d
    p = HERE / arm / f"layer_bits_folding_{arm}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fingerprint(d):
    items = sorted((n, v["pe"], v["simd"], v.get("thr_pe")) for n, v in d["per_layer"].items())
    items += sorted((n, v["pe"]) for n, v in d["extra_nodes"].items())
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()[:10]


def _ratchet_cfg(d):
    """MILP arm: _diagnostics.ratchet; analytical arm (net_fold.py): _diagnostics.analytical.{ratchet_pct, ratchet_floor} (its native rule is block to block; the edge check below is the MILP's node rule)."""
    g = d["_diagnostics"]
    if "ratchet" in g:
        return g["ratchet"]
    a = g["analytical"]
    return dict(pct=a["ratchet_pct"], floor=a["ratchet_floor"], n_constraints=None)


def ratchet_violations(d):
    """Edges (nearest compute ancestor -> compute node) that break cycles[C] <= max(floor * F, (1 + pct/100) * cycles[P]); same node set as finn_milp.py."""
    r = _ratchet_cfg(d)
    if r["pct"] is None:
        return None
    F = d["_diagnostics"]["max_node_cycles"]
    nodes = {**d["per_layer"], **d["extra_nodes"]}
    ops = {n for n in d["per_layer"] if not n.endswith(".pool")} | {n for n, v in d["extra_nodes"].items() if v["kind"] in ("argmax", "pad_mvau")}
    edges = d["dataflow_graph"]["edges"]
    memo = {}

    def anc(n):
        if n not in memo:
            out = set()
            for p in edges.get(n, []):
                if p in ops:
                    out.add(p)
                elif p in nodes or p in edges:
                    out |= anc(p)
            memo[n] = out
        return memo[n]
    bad = []
    for c in ops:
        for p in anc(c):
            if nodes[c]["cycles"] > max(r["floor"] * F, (1 + r["pct"] / 100) * nodes[p]["cycles"]) * (1 + 1e-6):
                bad.append((p, c))
    return bad


def main():
    rows, prints, groups = [], [], {}
    off = load("ratchet_off")
    for arm in ARMS:
        d = load(arm)
        if d is None:
            rows.append(dict(arm=arm, status="missing"))
            continue
        row = dict(arm=arm, status=d["status"])
        if d["status"] != "Optimal":
            rows.append(row)
            continue
        g = d["_diagnostics"]
        fm = g.get("fifo_model", {}).get("totals", {})
        inter = d.get("inter_block_fifos", [])
        if "fifo_model" in g:                                   # MILP arm: total_* already include the priced FIFOs and DWCs
            fifo_lut = fm.get("lut", 0) + fm.get("dwc_lut", 0) + sum(f["lut"] for f in inter)
            lut_total, bram_total, fifo_bram = g["total_lut_calibrated"], g["total_bram18k_calibrated"], fm.get("bram18", 0)
            n_dwcs, n_fifos = len(d.get("dwcs", [])), len(d.get("intra_block_fifos", []))
        else:                                                   # analytical arm: nodes only in total_*, simulated FIFOs / DWCs on top
            it = g["intra_block_fifo_totals"]
            fifo_lut = it["fifo_lut"] + it["dwc_lut"] + g["inter_block_fifo_lut"]
            lut_total, bram_total, fifo_bram = g["total_lut_calibrated"] + fifo_lut, g["total_bram18k_with_all_fifos"], it["fifo_bram18"]
            n_dwcs = sum(len(b["dwcs"]) for b in d["block_verification"].values())
            n_fifos = sum(1 for f in d.get("intra_block_fifos", []) if f["depth"] > 2)
        fp = fingerprint(d)
        groups.setdefault(fp if arm not in VARIANTS else f"{fp}:{arm}", []).append(arm)         # a variant always gets its own build
        bad = ratchet_violations(d)
        weights = {n: v["simd"] * v["weight_bits"] for n, v in d["per_layer"].items() if not n.endswith(".pool")}   # the analytical pad-MVAU (INT8 weights) is capped in its search too
        rc = _ratchet_cfg(d)
        row.update(
            ratchet_pct=rc["pct"], ratchet_floor=rc["floor"], ratchet_constraints=rc["n_constraints"],
            lut_total=round(lut_total), lut_nodes=round(lut_total - fifo_lut), lut_fifo_dwc=round(fifo_lut),
            bram18=bram_total, bram18_fifos=fifo_bram, dsp=g["total_dsp"],
            slowest_node=g["bottleneck_node"], slowest_cycles=g["bottleneck_cycles"], fps=round(1e8 / g["bottleneck_cycles"], 1),
            sum_node_cycles_ms=round(g["total_cycles"] / 1e5, 1), n_dwcs=n_dwcs, n_fifos_gt2=n_fifos,
            layers_diff_vs_off=(sum((v["pe"], v["simd"]) != (off["per_layer"][n]["pe"], off["per_layer"][n]["simd"]) for n, v in d["per_layer"].items())
                                if off and off["status"] == "Optimal" else None),
            fingerprint=fp, ratchet_violations=(None if bad is None else len(bad)), wwidth_max_seen=max(weights.values()),
            all_int6=all(v == BITS for v in d["layer_weight_bits"].values()) and all(v == BITS for v in d["layer_act_bits"].values()),
        )
        assert row["wwidth_max_seen"] <= WWIDTH_MAX, (arm, row["wwidth_max_seen"])
        if arm in VARIANTS:
            for k in ("lut_total", "lut_fifo_dwc", "bram18", "bram18_fifos", "n_dwcs", "n_fifos_gt2"):
                row[k] = None                                                                  # FINN autosizes the FIFOs: no model number
        rows.append(row)
    cols = ["arm", "status", "ratchet_pct", "ratchet_floor", "ratchet_constraints", "lut_total", "lut_nodes", "lut_fifo_dwc", "bram18", "bram18_fifos", "dsp", "slowest_node",
            "slowest_cycles", "fps", "sum_node_cycles_ms", "n_dwcs", "n_fifos_gt2", "layers_diff_vs_off", "fingerprint", "ratchet_violations", "wwidth_max_seen", "all_int6"]
    with open(HERE / "arms_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (HERE / "arm_groups.json").write_text(json.dumps(groups, indent=1))
    build = [arms[0] for arms in sorted(groups.values(), key=lambda a: ARMS.index(a[0]))]
    (HERE / "arms_to_build.txt").write_text("\n".join(build) + "\n")
    with open(HERE / "arm_build_map.csv", "w", newline="") as fh:         # arm -> the build that represents it (identical foldings share one build)
        w = csv.writer(fh)
        w.writerow(["arm", "build_arm"])
        for arms in groups.values():
            for arm in arms:
                w.writerow([arm, arms[0]])
    print(f"{'arm':14s}{'status':11s}{'LUT':>8s}{'(nodes)':>9s}{'BRAM18':>8s}{'DSP':>6s}{'slowest':>9s}{'fps':>7s}{'dLayers':>8s}  fingerprint  violations")
    for r in rows:
        if r["status"] != "Optimal":
            print(f"{r['arm']:14s}{r['status']}")
            continue
        if r["arm"] in VARIANTS:
            print(f"{r['arm']:14s}{r['status']:11s}{'-':>8s}{r['lut_nodes']:9d}{'-':>8s}{r['dsp']:6.0f}{r['slowest_cycles']:9d}{r['fps']:7.1f}{str(r['layers_diff_vs_off']):>8s}  {r['fingerprint']}  (same folding as {VARIANTS[r['arm']]}, FINN FIFO autosize)")
            continue
        print(f"{r['arm']:14s}{r['status']:11s}{r['lut_total']:8d}{r['lut_nodes']:9d}{r['bram18']:8.0f}{r['dsp']:6.0f}{r['slowest_cycles']:9d}{r['fps']:7.1f}"
              f"{str(r['layers_diff_vs_off']):>8s}  {r['fingerprint']}  {r['ratchet_violations']}")
    print("\nidentical foldings:", {fp: arms for fp, arms in groups.items()})
    print("arms to build (one per distinct folding):", build)


if __name__ == "__main__":
    main()
