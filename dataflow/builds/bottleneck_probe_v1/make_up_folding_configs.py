"""Folding + FIFO config (and the model's prediction) for the upsampling probes. Standard library only.

    python3 hardware/builds/bottleneck_probe_v1/make_up_folding_configs.py [--bits 4 6 8] [--variants conv noconv]

Same case as export_up_probe.py (32 -> 16, 32x32 input -> 64x64 output, T_out = 18 per output pixel). Writes
inputs/up_cin32_cout16_in32_int{b}_{variant}_folding.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "MILP" / "analytical"))

from up_bottleneck import model_up_bottleneck, to_folding_config, verify_with_sim  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "inputs"
BLOCKS = {"up4": (32, 16, 4, 32), "up5": (16, 4, 4, 64)}   # (Cin, Cout, ratio, input H=W), U4 widths
FRAME_CYCLES = 73728


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--variants", nargs="+", default=["conv", "noconv"], choices=["conv", "noconv"])
    ap.add_argument("--block", choices=sorted(BLOCKS), default="up4")
    a = ap.parse_args()
    CIN, COUT, V, HW_IN = BLOCKS[a.block]
    T_OUT = FRAME_CYCLES / (4 * HW_IN ** 2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for b in a.bits:
        for variant in a.variants:
            r = model_up_bottleneck(CIN, COUT, V, b, HW_IN, HW_IN, T_out=T_OUT, skip_conv=(variant == "conv"))
            verify_with_sim(r)
            name = f"up_cin{CIN}_cout{COUT}_in{HW_IN}_int{b}_{variant}"
            (OUT_DIR / f"{name}_folding.json").write_text(json.dumps(to_folding_config(r), indent=2))
            v = r.verification
            print(f"{name}: steady {v['steady_cyc_px']:.2f} cyc/px  skip FIFO {r.skip_fifo.depth_words} words  FIFO main {r.params['main_fifo_words']}  "
                  f"latency {v['latency_first_out']}  LUT {r.totals['lut']:.0f}  BRAM18 {r.totals['bram18']:.1f}  DSP {r.totals['dsp']:.0f}", flush=True)


if __name__ == "__main__":
    main()
