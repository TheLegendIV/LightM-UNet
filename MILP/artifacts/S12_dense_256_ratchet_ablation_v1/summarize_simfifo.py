"""Compare each MILP arm's FIFO estimate with its simulated FIFOs (run run_simfifo.sh first).

    python3 MILP/artifacts/S12_dense_256_ratchet_ablation_v1/summarize_simfifo.py

Writes next to this file: fifo_changes.csv (one row per FIFO and arm: MILP depth / BRAM18 / LUT where the MILP listed the FIFO, simulated ones next to it) and
fifo_sim_report.md (per arm: did the blocks / the chain deadlock or miss their rate, what grew, FIFO totals before and after, by FIFO class).
"""
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
MILP_ARMS = ["ratchet_1pct", "ratchet_25pct", "ratchet_100pct", "ratchet_200pct", "ratchet_off"]


def klass(f):
    p, c = f["producer"], f["consumer"]
    if f.get("is_skip"):
        return "skip FIFO"
    if c.endswith((".fmpad", ".fmpad_u")) or c.endswith("fmpadpix"):
        return "window prefetch (in front of FMPad)"
    if p.endswith(".dup"):
        return "dup output"
    if "dwc" in p or "dwc" in c:
        return "DWC side"
    return "other intra-block"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    rows, md = [], ["# FIFO simulation of the MILP arms (`net_explicit.py`)\n",
                    "Each MILP folding was fed unchanged into the analytical block models and simulators (per block: saturated input, 3 frames, budget = the block's slowest MILP node; "
                    "then the whole network chained with inter-block FIFOs at depth 2). FIFO depths are the smallest the simulation needs; a deadlocked chain grows the full FIFOs. "
                    "`MILP` = the closed-form estimate in the MILP json (skip + prefetch FIFOs only, DWC LUT); `sim` = every FIFO of the simulation.\n"]
    for arm in MILP_ARMS:
        base, sim = load(HERE / arm / f"layer_bits_folding_{arm}.json"), load(HERE / f"{arm}_simfifo" / f"layer_bits_folding_{arm}_simfifo.json")
        md.append(f"\n## {arm}\n")
        if base is None or sim is None:
            md.append("not run\n")
            continue
        fs = sim["fifo_sim"]
        g, gb = sim["_diagnostics"], base["_diagnostics"]
        est = g["fifo_model_estimate"]
        failed = {s: b for s, b in fs["blocks"].items() if not b["ok"]}
        missed = {s: b["rate_miss"] for s, b in fs["blocks"].items() if b.get("rate_miss")}
        miss_txt = (" (" + ", ".join("%s +%s%%" % (st, m["over_pct"]) for st, m in sorted(missed.items())) + ")") if missed else ""
        md.append(f"* blocks: {len(fs['blocks'])}; fail alone (cannot meet the network budget): **{len(failed)}** {sorted(failed)}; "
                  f"miss their own budget at any depth (window-fill gap, accepted if within the network budget): **{len(missed)}**"
                  + miss_txt)
        ch = fs["chain"]
        if ch is None:
            md.append("* whole-net chain: not run")
        else:
            first = ch["rounds"][0]
            md.append(f"* whole-net chain (inter-block depth 2, target {ch['target']} cycles/frame + {fs['slack']:.0%}): first round deadlock = **{first['deadlock']}** "
                      f"(period {first['period']}), rounds = {len(ch['rounds'])}, final ok = **{ch['ok']}**, last period {ch['last']['period']}")
            for r in ch["rounds"]:
                if not r["ok"]:
                    md.append(f"  * round {r['round']}: deadlock={r['deadlock']} period={r['period']}; full: {', '.join(r['full'][:8])}{' ...' if len(r['full']) > 8 else ''}; grew {len(r.get('grown', []))}")
            if ch["grown"]:
                md.append("  * grown by the chain: " + ", ".join(f"{x['where']}:{x['fifo']} {x['depth_before']}->{x['depth_after']}" for x in ch["grown"]))
            if any(d > 2 for d in ch["interface_depths"]):
                md.append("  * inter-block depths > 2: " + ", ".join(f"#{i}={d}" for i, d in enumerate(ch["interface_depths"]) if d > 2))
        before, after = {}, {}
        for f in base.get("intra_block_fifos", []):
            before[(f["stage"], f["producer"], f["consumer"])] = f
        for f in sim["intra_block_fifos"]:
            after[(f["stage"], f["producer"], f["consumer"])] = f
        for key in sorted(before.keys() | after.keys()):
            b, s = before.get(key), after.get(key)
            ref = s or b
            rows.append(dict(arm=arm, stage=key[0], klass=klass(ref), producer=key[1], consumer=key[2], width_bits=ref["width_bits"], in_milp_list=b is not None, in_sim=s is not None,
                             depth_milp=(b or {}).get("depth"), depth_sim=(s or {}).get("depth"), bram18_milp=(b or {}).get("mem_bram18"), bram18_sim=(s or {}).get("mem_bram18"),
                             lut_milp=(b or {}).get("mem_lut"), lut_sim=(s or {}).get("mem_lut"), mem_sim=(s or {}).get("mem")))
        mine = [r for r in rows if r["arm"] == arm]
        md.append("\n| class | FIFOs in sim | listed by MILP | MILP BRAM18 | sim BRAM18 | listed: median depth MILP -> sim | listed: depth up / down / same | not listed by MILP: sim depth > 2 | their BRAM18 |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for k in sorted({r["klass"] for r in mine}):
            c = [r for r in mine if r["klass"] == k]
            both = [r for r in c if r["in_milp_list"] and r["in_sim"]]
            unl = [r for r in c if not r["in_milp_list"] and r["in_sim"]]
            up = sum(r["depth_sim"] > r["depth_milp"] for r in both)
            dn = sum(r["depth_sim"] < r["depth_milp"] for r in both)
            med = f"{statistics.median(r['depth_milp'] for r in both):.0f} -> {statistics.median(r['depth_sim'] for r in both):.0f}" if both else "-"
            md.append(f"| {k} | {sum(r['in_sim'] for r in c)} | {sum(r['in_milp_list'] for r in c)} | {sum(r['bram18_milp'] or 0 for r in c)} | {sum(r['bram18_sim'] or 0 for r in c)} | {med} | "
                      f"{up} / {dn} / {len(both) - up - dn} | {sum(r['depth_sim'] > 2 for r in unl)} | {sum(r['bram18_sim'] or 0 for r in unl)} |")
        md.append("")
        big = sorted((r for r in mine if r["in_sim"]), key=lambda r: -(r["bram18_sim"] or 0) - (r["depth_sim"] or 0) / 1e6)[:8]
        md.append("largest simulated FIFOs: " + "; ".join(f"{r['stage']} {r['producer'].split('.', 1)[-1]}->{r['consumer'].split('.', 1)[-1]} {r['depth_milp']}->{r['depth_sim']} ({r['bram18_sim']} BRAM18)" for r in big))
        ib_old = base.get("inter_block_fifos", [])
        md.append(f"\n* inter-block FIFOs: {len(ib_old)} at depth {sorted({f['depth'] for f in ib_old})} (MILP) -> depths {sorted({f['depth'] for f in sim['inter_block_fifos']})} (sim), "
                  f"{g['inter_block_fifo_bram18']} BRAM18")
        md.append(f"* intra-block FIFO + DWC totals: MILP estimate {est['bram18']} BRAM18 / {est['lut']:.0f} LUT (skip + prefetch + DWC + inter) -> sim "
                  f"{g['intra_block_fifo_bram18'] + g['inter_block_fifo_bram18']} BRAM18 / {g['total_lut_with_all_fifos'] - g['total_lut_calibrated']:.0f} LUT; "
                  f"design total BRAM18 {gb['total_bram18k_calibrated']} -> {g['total_bram18k_with_all_fifos']}, LUT {gb['total_lut_calibrated']:.0f} -> {g['total_lut_with_all_fifos']:.0f}\n")
    with open(HERE / "fifo_changes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["arm"])
        w.writeheader()
        w.writerows(rows)
    (HERE / "fifo_sim_report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
