"""FIFO sizes BEFORE simulation: the analytical flow's closed-form estimates vs the MILP's (run run_compare.sh first for the MILP solve).

    python3 MILP/artifacts/S12_dense_256_fifo_compare_v1/summarize_compare.py        (in lightmunet_dev)

Same target for both flows: 256x256 S12 dense nearest-upsample ReLU, uniform INT6, 250 fps (every node <= 400,000 cycles/frame), no MVAU width cap, default rate rule (--dsr-pct 4 --dsr-floor 0.6).
  MILP        milp/layer_bits_folding_milp.json: the MILP's closed-form lists for the MILP folding (skip + prefetch FIFOs, DWCs)
  ANALYTICAL  the analytical models' own estimates for the analytical folding, taken straight from the block models (bottleneck.py ... up_bottleneck.py) BEFORE verify_with_sim:
              skip FIFO = latency estimate (r.skip_fifo), up / initial `FIFO main` = r.params['main_fifo_words'], prefetch = (pad*W + pad + 1)*cf + 2 words (the formula verify_with_sim starts from),
              DWCs = r.dwcs. No cycle simulation anywhere; the analytical folding is rebuilt with net_fold.assemble (no verification).
Writes presim_analytical.json, fifo_compare.csv and fifo_compare.md next to this file.
"""
import csv
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO / "enet", REPO / "MILP", REPO / "MILP" / "utils", REPO / "MILP" / "analytical"):
    sys.path.insert(0, str(p))

import finn_milp  # noqa: E402
import net_fold  # noqa: E402
from bottleneck import fifo_memory  # noqa: E402
from fifo_model import prefetch_depth_words  # noqa: E402

BITS = 6


def analytical_presim():
    finn_milp.load_config("config_12_dense_relu_nearest_upsample_256")
    finn_milp.CANDIDATE_BITS = tuple(sorted(set(finn_milp.CANDIDATE_BITS) | {BITS}))
    _m, geoms, extras, _p, dmap, kinds = finn_milp.build_model_and_graph()
    geom = {g.name: g for g in geoms}
    ctx = (geoms, extras, geom, {n.geom.name: n for n in extras}, dmap, kinds)
    a = types.SimpleNamespace(bits=BITS, fps=250.0, clock_mhz=100.0, max_latency_ms=None, dsr_pct=4.0, dsr_floor=0.6, no_dsr=False, mvau_wwidth_max=None)
    res = net_fold.assemble(a, ctx, int(100e6 / 250), keep_blocks=True)
    fifos, dwcs = [], []

    def add(stage, klass, name, producer, consumer, depth, width, cout=None, pe=None):
        m = fifo_memory(width, depth)
        fifos.append(dict(stage=stage, klass=klass, name=name, producer=producer, consumer=consumer, depth=int(depth), width_bits=int(width), mem=m["mem"],
                          bram18=int(m["bram18"]), uram=int(m["uram"]), lut=int(m["lut"]), words_per_px=(cout // pe if cout and pe else None)))

    for stage, kind, r in res["_blocks"]:
        nodes = {n.name: n for n in r.nodes}
        cout = r.params["cout"]
        if kind == "init":
            sf = r.skip_fifo
            add(stage, "skip FIFO", f"{stage}.skip FIFO", f"{stage}.pool_quant", f"{stage}.concat", sf.depth_words, sf.width_bits, cout, sf.pe)
            add(stage, "FIFO main", f"{stage}.FIFO main", f"{stage}.conv", f"{stage}.concat", r.params["main_fifo_words"], BITS * nodes["Thr_c"].pe, cout, nodes["Thr_c"].pe)
        elif kind in ("reg", "dn"):
            sf = r.skip_fifo
            add(stage, "skip FIFO", f"{stage}.skip FIFO", f"{stage}.thr_s", f"{stage}.add", sf.depth_words, sf.width_bits, cout, sf.pe)
        elif kind == "up":
            sf = r.skip_fifo
            add(stage, "skip FIFO", f"{stage}.skip FIFO", f"{stage}.thr_e", f"{stage}.add", sf.depth_words, sf.width_bits, cout, sf.pe)
            add(stage, "FIFO main", f"{stage}.FIFO main", f"{stage}.thr_s", f"{stage}.add", r.params["main_fifo_words"], BITS * nodes["Thr_s"].pe, cout, nodes["Thr_s"].pe)
        for d in r.dwcs:
            dwcs.append(dict(stage=stage, name=d.edge, in_width=d.in_width, out_width=d.out_width, lut=float(d.lut)))
    for name, g in geom.items():                    # prefetch: the analytical sizing starts from the same formula the MILP uses
        if g.op_type == "MaxPool2d" or g.kh * g.kw == 1 or not (g.ph or g.pw):
            continue
        swu = res["per_layer"][name]["simd_swu"]
        add(g.stage, "window prefetch", f"{name}.prefetch", name, f"{name}.fmpad", prefetch_depth_words(g, swu), swu * BITS)
    folds = {n: (v["pe"], v["simd"]) for n, v in res["per_layer"].items()}
    return dict(fifos=fifos, dwcs=dwcs, folds=folds, dsp=res["_diagnostics"]["total_dsp"], node_lut=res["_diagnostics"]["total_lut_calibrated"],
                slowest=(res["_diagnostics"]["bottleneck_node"], res["_diagnostics"]["bottleneck_cycles"]))


def milp_lists(d):
    out = []
    for f in d.get("intra_block_fifos", []):
        kl = "skip FIFO" if f.get("is_skip") else ("window prefetch" if f["name"].endswith(".prefetch") else ("FIFO main" if f["name"].endswith(".FIFO_main") else "other"))
        out.append(dict(stage=f["stage"], klass=kl, name=f["name"], depth=f["depth"], width_bits=f["width_bits"], mem=f.get("mem"), bram18=f.get("mem_bram18", 0), uram=f.get("mem_uram18", 0), lut=f.get("mem_lut", 0)))
    return out


def main():
    milp = json.loads((HERE / "milp/layer_bits_folding_milp.json").read_text())
    pre = analytical_presim()
    (HERE / "presim_analytical.json").write_text(json.dumps(pre, indent=1))
    mf = milp_lists(milp)
    g = milp["_diagnostics"]
    ft = g["fifo_model"]["totals"]
    md = ["# FIFO sizes before simulation: analytical flow vs MILP\n",
          "256x256 S12 dense nearest-upsample ReLU, uniform INT6, 250 fps (every node <= 400,000 cycles/frame), no MVAU width cap, default rate rule (--dsr-pct 4 --dsr-floor 0.6). "
          "ANALYTICAL = the block models' own closed-form estimates (no cycle simulation); MILP = `fifo_model.py` priced in the solve. Each flow keeps its own folding.\n"]

    def tot(lst, klass=None):
        sel = [f for f in lst if klass is None or f["klass"] == klass]
        return len(sel), sum(f["bram18"] for f in sel), sum(f["lut"] for f in sel), sum(f.get("uram", 0) for f in sel)
    md.append("## Designs\n")
    md.append("| flow | DSP | slowest node (cycles) | fps |")
    md.append("|---|---|---|---|")
    md.append(f"| MILP | {g['total_dsp']:.0f} | {g['bottleneck_cycles']} ({g['bottleneck_node']}) | {1e8 / g['bottleneck_cycles']:.0f} |")
    md.append(f"| ANALYTICAL | {pre['dsp']:.0f} | {pre['slowest'][1]} ({pre['slowest'][0]}) | {1e8 / pre['slowest'][1]:.0f} |\n")

    md.append("## Totals by FIFO class (count / BRAM18 / LUT / URAM288 of the FIFO memory)\n")
    md.append("| class | MILP | ANALYTICAL |")
    md.append("|---|---|---|")
    for kl in ("skip FIFO", "FIFO main", "window prefetch"):
        m, a_ = tot(mf, kl), tot(pre["fifos"], kl)
        md.append(f"| {kl}{' (up, initial)' if kl == 'FIFO main' else ''} | {m[0]} / {m[1]} / {m[2]} / {m[3]} | {a_[0]} / {a_[1]} / {a_[2]} / {a_[3]} |")
    m_dwc, a_dwc = len(milp.get("dwcs", [])), len(pre["dwcs"])
    md.append(f"| DWCs | {m_dwc} / - / {ft['dwc_lut']:.0f} | {a_dwc} / - / {sum(d['lut'] for d in pre['dwcs']):.0f} |")
    m_all, a_all = tot(mf), tot(pre["fifos"])
    md.append(f"| **FIFO memory + DWC LUT** | {m_all[1]} BRAM18, {m_all[3]} URAM, {m_all[2] + ft['dwc_lut']:.0f} LUT | {a_all[1]} BRAM18, {a_all[3]} URAM, {a_all[2] + sum(d['lut'] for d in pre['dwcs']):.0f} LUT |\n")

    def by(lst, kl):
        return {f["stage"]: f for f in lst if f["klass"] == kl}
    md.append("## Skip FIFOs: depth in words (and pixels) at each flow's own folding\n")
    md.append("| block | MILP words (px, width, BRAM18) | ANALYTICAL words (px, width, BRAM18) | analytical px / MILP px | ANALYTICAL px at the MILP's width: BRAM18 |")
    md.append("|---|---|---|---|---|")
    ms, as_ = by(mf, "skip FIFO"), by(pre["fifos"], "skip FIFO")
    rows, both = [], []
    for stage in sorted(set(ms) | set(as_), key=lambda s: next(i for i, f in enumerate(pre["fifos"]) if f["stage"] == s) if s in as_ else 10 ** 6):
        m, a_ = ms.get(stage), as_.get(stage)
        apx = None if a_ is None else a_["depth"] // a_["words_per_px"]
        mpx = None
        if m is not None and a_ is not None:                      # Cout is the same for both; MILP words per pixel = Cout / (width / bits)
            mpx = m["depth"] // (a_["words_per_px"] * (a_["width_bits"] // BITS) // (m["width_bits"] // BITS))
        a_at_m = None
        if m is not None and a_ is not None:       # the analytical pixel estimate at the MILP's skip-node PE: separates "different PE" from "different depth estimate"
            cout = a_["words_per_px"] * (a_["width_bits"] // BITS)
            a_at_m = fifo_memory(m["width_bits"], apx * (cout // (m["width_bits"] // BITS)))["bram18"]
            both.append((m["bram18"], a_["bram18"], a_at_m))
        md.append(f"| {stage} | {'-' if m is None else str(m['depth']) + ' (' + str(mpx) + ' px, ' + str(m['width_bits']) + ' b, ' + str(m['bram18']) + ')'} | "
                  f"{'-' if a_ is None else str(a_['depth']) + ' (' + str(apx) + ' px, ' + str(a_['width_bits']) + ' b, ' + str(a_['bram18']) + ')'} | "
                  f"{'-' if (m is None or a_ is None) else format(apx / mpx, '.2f') + 'x'} | {'-' if a_at_m is None else a_at_m} |")
        rows.append(dict(stage=stage, milp_depth=None if m is None else m["depth"], milp_width=None if m is None else m["width_bits"], milp_px=mpx,
                         analytical_depth=None if a_ is None else a_["depth"], analytical_width=None if a_ is None else a_["width_bits"], analytical_px=apx))
    md.append("")
    md.append(f"Skip FIFOs both flows price ({len(both)} blocks): MILP {sum(b[0] for b in both)} BRAM18; ANALYTICAL {sum(b[1] for b in both)} BRAM18 at its own PEs, "
              f"{sum(b[2] for b in both)} BRAM18 with its depth estimate at the MILP's PEs. (Last two differ by the PE choice: stream width -> BRAM aspect / pow2 rounding; "
              f"MILP vs the last one differ by the depth estimate.)\n")
    sim = {}
    v2 = HERE.parents[1] / "artifacts/S12_dense_256_u4_analytical_v2/int6_fps250_lat200/layer_bits_folding_final.json"
    if v2.exists():                                           # the simulated reference (real row-buffer UpsampleNearestNeighbour) at the analytical folding
        for f in json.loads(v2.read_text())["intra_block_fifos"]:
            if f["stage"] in ("up4", "up5") and (f["name"].endswith("FIFO_main") or f["is_skip"]):
                sim[(f["stage"], "skip FIFO" if f["is_skip"] else "FIFO main")] = f["depth"]
    md.append("## Up blocks: the two join FIFOs (words)\n")
    md.append("| block | FIFO | MILP closed form | ANALYTICAL pre-simulation | simulation (analytical_v2, real UpsampleNN kernel) |")
    md.append("|---|---|---|---|---|")
    mm, am = by(mf, "FIFO main"), by(pre["fifos"], "FIFO main")
    for stage in ("up4", "up5"):
        for kl in ("skip FIFO", "FIFO main"):
            m = (ms if kl == "skip FIFO" else mm).get(stage)
            a_ = (as_ if kl == "skip FIFO" else am).get(stage)
            md.append(f"| {stage} | {kl} | {'-' if m is None else str(m['depth']) + ' (' + str(m['width_bits']) + ' b)'} | {'-' if a_ is None else a_['depth']} | {sim.get((stage, kl), '-')} |")
    init = am.get("initial")
    md.append(f"\n(initial block `FIFO main`: analytical estimate {init['depth'] if init else '-'} words, 3-4 words in simulation; not priced by the MILP.)\n")
    pm = {f["stage"]: f for f in mf if f["klass"] == "window prefetch"}
    pa = {f["stage"]: f for f in pre["fifos"] if f["klass"] == "window prefetch"}
    same = sum(1 for k in pm if k in pa and pm[k]["depth"] == pa[k]["depth"])
    md.append(f"## Prefetch FIFOs\n\n{len(pm)} listed by the MILP, {len(pa)} by the analytical estimate; the same formula `(pad*W + pad + 1) * cf + 2` words, cf from each flow's own SWG SIMD; "
              f"{same} of {len(pm)} have identical depth at the two foldings (BRAM18 {sum(f['bram18'] for f in pm.values())} vs {sum(f['bram18'] for f in pa.values())}).\n")
    with open(HERE / "fifo_compare.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (HERE / "fifo_compare.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
