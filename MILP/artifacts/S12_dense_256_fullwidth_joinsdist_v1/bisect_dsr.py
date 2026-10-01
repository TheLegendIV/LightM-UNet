"""Min feasible pass-2 DSR (to 0.01) for S12 dense 256x256 full width (config_12_dense_relu_nearest_conv_upsample_256).
Pass 1 (accuracy, bits 4/6/8, LUT 0.7 BRAM 0.4 DSP 0.9, no FPS/DSR, --joins-distributed) runs once; every probe is pass 2
as a standalone solve: bits pinned to pass 1 (--pin-bits-file), --min-resources, same caps, --target-fps 100, --dsr-ratio X.
That is exactly what --lexicographic's pass 2 solves when --dsr-ratio-pass2 is given (roof = hard caps). Feasibility is
monotone in DSR, so a k-ary search on integer hundredths finds the smallest feasible ratio.
Run inside lightmunet_dev from /workspace/LightM-UNet:  python3 MILP/artifacts/S12_dense_256_fullwidth_joinsdist_v1/bisect_dsr.py
"""
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path("MILP/artifacts/S12_dense_256_fullwidth_joinsdist_v1")
COMMON = ["python3", "MILP/finn_milp.py", "--config", "config_12_dense_relu_nearest_conv_upsample_256",
          "--sensitivity-file", "MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_conv_upsample_256.json",
          "--candidate-bits", "4,6,8", "--hard-lut-fraction", "0.7", "--hard-bram-fraction", "0.4",
          "--hard-dsp-fraction", "0.9", "--force-dsp", "--joins-distributed", "--time-limit", "600", "--gap-rel", "0.005"]
WORKERS = 8


def sh(cmd, log):
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)


def status(path):
    p = Path(path)
    return json.loads(p.read_text())["status"] if p.exists() else "missing"


def pass1():
    d = OUT / "pass1"
    d.mkdir(parents=True, exist_ok=True)
    out = d / "layer_bits_folding_pass1.json"
    sh(COMMON + ["--out-file", str(out)], d / "solve.log")  # pass 1 only: accuracy objective, no --lexicographic
    assert status(out) == "Optimal", f"pass 1 status {status(out)}"
    return out


def probe(pins, hundredths):
    d = OUT / "probes" / f"dsr_{hundredths / 100:.2f}"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"layer_bits_folding_dsr{hundredths}.json"
    sh(COMMON + ["--pin-bits-file", str(pins), "--min-resources", "--target-fps", "100",
                 "--dsr-ratio", f"{hundredths / 100:.2f}", "--out-file", str(out)], d / "solve.log")
    return hundredths, status(out)


def main():
    pins = pass1()
    print("pass 1 done:", pins, flush=True)
    results = {}
    lo, hi = 100, None  # lo = largest known infeasible (1.00 assumed infeasible, verified below), hi = smallest known feasible
    grid = [101, 102, 105, 110, 125, 150, 200, 300, 500, 800]
    with ThreadPoolExecutor(WORKERS) as ex:
        for h, st in ex.map(lambda x: probe(pins, x), grid):
            results[h] = st
            print(f"DSR {h / 100:.2f}: {st}", flush=True)
        feas = sorted(h for h, st in results.items() if st == "Optimal")
        if not feas:
            print("no feasible DSR up to 8.00 -- stop"); return
        hi = feas[0]
        infeas = [h for h, st in results.items() if st == "Infeasible" and h < hi]
        lo = max(infeas) if infeas else 99
        while hi - lo > 1:
            k = min(WORKERS - 1, hi - lo - 1)
            pts = sorted({lo + round((hi - lo) * (i + 1) / (k + 1)) for i in range(k)} - {lo, hi})
            for h, st in ex.map(lambda x: probe(pins, x), pts):
                results[h] = st
                print(f"DSR {h / 100:.2f}: {st}", flush=True)
            feas = [h for h in pts if results[h] == "Optimal"]
            infe = [h for h in pts if results[h] == "Infeasible"]
            if feas:
                hi = min(feas)
            if infe:
                lo = max(infe)
    (OUT / "min_feasible_dsr.json").write_text(json.dumps({"min_feasible_dsr": hi / 100, "largest_infeasible": lo / 100,
                                                          "results": {f"{k / 100:.2f}": v for k, v in sorted(results.items())}}, indent=1))
    print(f"MIN FEASIBLE DSR = {hi / 100:.2f} (largest infeasible probed {lo / 100:.2f})", flush=True)


if __name__ == "__main__":
    main()
