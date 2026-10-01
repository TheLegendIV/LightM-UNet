"""Min feasible DSR (to 0.01) with FPS 100 and DSR applied in BOTH passes, S12 dense 256x256 full width
(config_12_dense_relu_nearest_conv_upsample_256, bits 4/6/8, LUT 0.7 BRAM 0.4 DSP 0.9, --joins-distributed, force-dsp).
Each probe is a full pass 1 (accuracy objective) at --dsr-ratio X --target-fps 100: pass 2 only re-folds under pass 1's own
use, so it is feasible whenever pass 1 is. Feasibility is monotone in X (bits are free, so this is the loosest case).
Then run:  finn_milp.py ... --lexicographic --target-fps 100 --dsr-ratio <min>  for the deliverable.
Run inside lightmunet_dev from /workspace/LightM-UNet.
"""
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path("MILP/artifacts/S12_dense_256_fullwidth_joinsdist_v1/search_pass1")
COMMON = ["python3", "MILP/finn_milp.py", "--config", "config_12_dense_relu_nearest_conv_upsample_256",
          "--sensitivity-file", "MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_conv_upsample_256.json",
          "--candidate-bits", "4,6,8", "--hard-lut-fraction", "0.7", "--hard-bram-fraction", "0.4",
          "--hard-dsp-fraction", "0.9", "--force-dsp", "--joins-distributed", "--target-fps", "100",
          "--time-limit", "600", "--gap-rel", "0.005"]
WORKERS = 8


def probe(h):
    d = OUT / f"dsr_{h / 100:.2f}"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"layer_bits_folding_dsr{h}.json"
    with open(d / "solve.log", "w") as f:
        subprocess.run(COMMON + ["--dsr-ratio", f"{h / 100:.2f}", "--out-file", str(out)], stdout=f, stderr=subprocess.STDOUT)
    return h, (json.loads(out.read_text())["status"] if out.exists() else "missing")


def main():
    results = {}
    with ThreadPoolExecutor(WORKERS) as ex:
        grid = [101, 102, 105, 110, 125, 150, 200, 300, 500, 800, 1200, 2000]
        for h, st in ex.map(probe, grid):
            results[h] = st
            print(f"DSR {h / 100:.2f}: {st}", flush=True)
        feas = sorted(h for h, st in results.items() if st == "Optimal")
        if not feas:
            print("no feasible DSR up to 20.00 -- stop", flush=True)
            return
        hi = feas[0]
        infeas = [h for h, st in results.items() if st == "Infeasible" and h < hi]
        lo = max(infeas) if infeas else 99
        while hi - lo > 1:
            k = min(WORKERS - 1, hi - lo - 1)
            pts = sorted({lo + round((hi - lo) * (i + 1) / (k + 1)) for i in range(k)} - {lo, hi})
            for h, st in ex.map(probe, pts):
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
