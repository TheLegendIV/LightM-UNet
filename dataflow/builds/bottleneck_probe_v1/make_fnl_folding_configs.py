"""Folding + FIFO config (and the model's prediction) for the final-deconvolution probes. Standard library only.

    python3 hardware/builds/bottleneck_probe_v1/make_fnl_folding_configs.py [--bits 4 6 8] [--variants bias nobias]

Same case as export_fnl_probe.py (4 -> 5 channels, 128x128 -> 256x256, F = 73728). Writes inputs/fnl_cin4_cout5_in128_int{b}_{variant}_folding.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "MILP" / "analytical"))

from fnl_block import model_fnl_block, to_folding_config, verify_with_sim  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "inputs"
CIN, COUT, HW_IN, F = 4, 5, 128, 73728


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--variants", nargs="+", default=["bias", "nobias"], choices=["bias", "nobias"])
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for b in a.bits:
        for variant in a.variants:
            r = model_fnl_block(CIN, COUT, b, HW_IN, HW_IN, F=F, bias=(variant == "bias"))
            verify_with_sim(r)
            name = f"fnl_cin{CIN}_cout{COUT}_in{HW_IN}_int{b}_{variant}"
            (OUT_DIR / f"{name}_folding.json").write_text(json.dumps(to_folding_config(r), indent=2))
            v = r.verification
            print(f"{name}: steady {v['steady_cyc_px']:.2f} cyc/px  latency {v['latency_first_out']}  "
                  f"LUT {r.totals['lut']:.0f}  BRAM18 {r.totals['bram18']:.1f}  DSP {r.totals['dsp']:.0f}", flush=True)


if __name__ == "__main__":
    main()
