"""Generate the role-keyed FINN folding + FIFO config (and the model's prediction) for every probe case.

Pure Python (standard library only): runs on the host or in lightmunet_dev. Uses MILP/analytical/bottleneck.py, so each case is
model_bottleneck -> verify_with_sim (sizes FIFOs, proves no deadlock in simulation) -> to_folding_config.

    python3 hardware/builds/bottleneck_probe_v1/make_folding_configs.py [--dilations ...] [--bits ...] [--T 72]

Writes inputs/bottleneck_cin32_d{d}_int{b}_folding.json (see to_folding_config for the schema).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "MILP" / "analytical"))

from bottleneck import model_bottleneck, to_folding_config, verify_with_sim  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "inputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--T", type=int, default=72, help="target cycles per output pixel")
    ap.add_argument("--cin", type=int, default=32)
    ap.add_argument("--size", type=int, default=32)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for d in a.dilations:
        for b in a.bits:
            r = model_bottleneck(a.cin, 4, 4, a.T, b, a.size, a.size, 3, d, 1)
            verify_with_sim(r)
            cfg = to_folding_config(r)
            name = f"bottleneck_cin{a.cin}_d{d}_int{b}"
            (OUT_DIR / f"{name}_folding.json").write_text(json.dumps(cfg, indent=2))
            v = r.verification
            print(f"{name}: steady {v['steady_cyc_px']:.2f} cyc/px  skip FIFO {r.skip_fifo.depth_words} words  "
                  f"latency {v['latency_first_out']}  LUT {r.totals['lut']:.0f}", flush=True)


if __name__ == "__main__":
    main()
