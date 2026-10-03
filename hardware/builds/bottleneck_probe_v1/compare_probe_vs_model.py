"""Compare landed hardware results of the bottleneck probes with the analytical model + simulation (host, std-lib only).

    python3 compare_probe_vs_model.py [results_dir]        # default: ./results (see collect_probe_outputs.sh)

Per probe_result.json prints: verdict (PASS = rtlsim finished AND steady cyc/px <= T*1.02), rtlsim steady vs target vs
simulated cyc/px, rtlsim vs predicted first-out latency, folding landed vs predicted (PE/SIMD per role), forced FIFO
depths, and (if --ooc was run) LUT / BRAM18 / DSP landed vs predicted. Exit code 1 if any probe fails the verdict.
"""
import glob
import json
import os
import sys

TOL = 1.02


def verdict(res):
    rs = res.get("stages", {}).get("rtlsim")
    if not rs:
        return "NO-RTLSIM", None
    cpp = rs.get("steady_cyc_per_pixel")
    if cpp is None:
        return "NO-THROUGHPUT", None
    return ("PASS" if cpp <= res["predicted"]["T"] * TOL else "SLOW"), cpp


def landed_vs_pred(res):
    landed = res.get("stages", {}).get("folding", {})
    pred = res["predicted"]["nodes"]
    bad = []
    for role, p in pred.items():
        l = landed.get(role, {})
        for key, pk in (("PE", "pe"), ("SIMD", "simd")):
            if key in l and pk in p and p[pk] and l[key] != p[pk]:
                bad.append(f"{role}.{key} landed {l[key]} != predicted {p[pk]}")
    return bad


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    files = sorted(glob.glob(os.path.join(root, "*", "probe_result.json")))
    if not files:
        print(f"no probe_result.json under {root}")
        sys.exit(2)
    failed = 0
    print(f"{'case':36s} {'verdict':10s} {'rtlsim':>8s} {'T':>4s} {'sim':>7s}  {'lat rtl/pred':>16s}  LUT land/pred   BRAM land/pred  DSP land/pred")
    for f in files:
        res = json.load(open(f))
        v, cpp = verdict(res)
        p = res["predicted"]
        rs = res.get("stages", {}).get("rtlsim", {})
        lat = rs.get("latency_cycles") or rs.get("latency[cycles]") or rs.get("cycles")
        ooc = res.get("stages", {}).get("ooc", {})
        tot = p["totals"]
        line = (f"{os.path.basename(os.path.dirname(f)):36s} {v:10s} "
                f"{(f'{cpp:.2f}' if cpp else '-'):>8s} {p['T']:>4d} {p['steady_cyc_px']:>7.2f}  "
                f"{str(lat):>8s}/{str(p['latency_first_out']):>7s}")
        if ooc:
            line += (f"  {ooc.get('LUT', '-')}/{tot['lut']:.0f}  {ooc.get('BRAM_18K', '-')}/{tot['bram18']:.1f}  "
                     f"{ooc.get('DSP', '-')}/{tot['dsp']:.0f}")
        print(line)
        for msg in landed_vs_pred(res):
            print("    folding mismatch:", msg)
        forced = [x for x in res.get("stages", {}).get("fifo", []) if x.get("forced_depth") is not None]
        skip = [x for x in forced if x.get("is_skip")]
        if skip:
            print(f"    skip FIFO forced {skip[0]['forced_depth']} words (stock {skip[0]['stock_depth']}); "
                  f"{len(forced)} FIFOs forced, {len(res['stages']['fifo']) - len(forced)} left at stock")
        if v != "PASS":
            failed += 1
    print(f"\n{len(files) - failed}/{len(files)} PASS (steady cyc/px <= T*{TOL})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
