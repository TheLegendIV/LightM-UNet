"""Compare landed hardware results of the bottleneck probes with the analytical model + simulation (host, std-lib only).

    python3 compare_probe_vs_model.py [results_dir]        # default: ./results (see collect_probe_outputs.sh)

Steady throughput is RECOMPUTED here from the raw rtlsim numbers:
    cyc/px = (cycles_N_frames - cycles_1_frame) / (N - 1) / pixels_per_frame
FINN's own "stable_throughput[images/s]" divides (cycles - latency) by N instead of N-1 (the single-frame run already contains one
frame), so it under-reports the period by (N-1)/N: 6 frames -> 17% too fast (hardware d1/d2/d8 INT4: FINN says 61.4 / 63.2 / 58.6,
the true values are 73.7 / 75.8 / 87.9). That field is printed in column "FINN" only for reference.

Verdict: PASS = rtlsim finished and steady <= T*1.02; NEAR = within the 3% the model accepts as inherent per-frame window-fill gap;
SLOW otherwise. Prints rtlsim vs model-simulation steady, one-frame cycles (rtlsim vs predicted), forced FIFO depths and, if --ooc
was run, LUT / BRAM18 / DSP landed vs predicted. Exit code 1 if any probe is SLOW or has no rtlsim.
"""
import glob
import json
import os
import sys

TOL, TOL_NEAR = 1.02, 1.03


def pixels_per_frame(res):
    p = res["probe"]
    if p.get("block") in ("down", "up", "init", "final"):
        return p["hout"] * p["wout"]
    return p["height"] * p["width"]


def steady(res):
    """(true cyc/px, FINN's reported cyc/px or None)."""
    rs = res.get("stages", {}).get("rtlsim")
    if not rs or "cycles" not in rs or "latency_cycles" not in rs or rs.get("N", 0) < 2:
        return None, None
    px = pixels_per_frame(res)
    true = (rs["cycles"] - rs["latency_cycles"]) / (rs["N"] - 1) / px
    finn = None
    if rs.get("stable_throughput[images/s]"):
        finn = rs.get("fclk[mhz]", 100.0) * 1e6 / rs["stable_throughput[images/s]"] / px
    return true, finn


def verdict(res):
    true, _ = steady(res)
    if true is None:
        return "NO-RTLSIM", None
    T = res["predicted"]["T"]
    return ("PASS" if true <= T * TOL else "NEAR" if true <= T * TOL_NEAR else "SLOW"), true


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
    print(f"{'case':52s} {'verdict':9s} {'rtlsim':>7s} {'FINN':>6s} {'T':>5s} {'model':>7s}  {'1-frame cyc rtl/pred':>22s}  LUT land/pred  BRAM land/pred  DSP land/pred")
    for f in files:
        res = json.load(open(f))
        v, true = verdict(res)
        _, finn = steady(res)
        p = res["predicted"]
        rs = res.get("stages", {}).get("rtlsim", {})
        ooc = res.get("stages", {}).get("ooc", {})
        tot = p["totals"]
        frame_pred = p.get("frame_cycles")
        line = (f"{os.path.basename(os.path.dirname(f)):52s} {v:9s} {(f'{true:.2f}' if true else '-'):>7s} "
                f"{(f'{finn:.1f}' if finn else '-'):>6s} {p['T']:>5.1f} {p['steady_cyc_px']:>7.2f}  "
                f"{str(rs.get('latency_cycles')):>10s}/{str(frame_pred):>10s}")
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
        if v in ("SLOW", "NO-RTLSIM"):
            failed += 1
    print(f"\n{len(files) - failed}/{len(files)} not SLOW (PASS <= T*{TOL}, NEAR <= T*{TOL_NEAR}); steady = (cycles - 1-frame cycles) / (N-1) / px")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
