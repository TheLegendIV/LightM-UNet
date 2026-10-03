"""Folding + FIFO config (and the model's prediction) for the initial-block probes. Standard library only.

    python3 hardware/builds/bottleneck_probe_v1/make_int_folding_configs.py [--bits 4 6 8]

Same case as export_int_probe.py (1 -> 4 channels, 256x256 -> 128x128, F = 81920). Writes inputs/init_cin1_cout4_in256_int{b}_folding.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "MILP" / "analytical"))

from int_bottleneck import model_int_bottleneck, to_folding_config, verify_with_sim  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "inputs"
CIN, COUT, HW_IN, F = 1, 4, 256, 81920


def _pool_case(base: str, name: str, extra: dict) -> None:
    """The Pool-route probe shares the ONNX with its StreamingMaxPool twin (the difference is FINN's conversion of the MaxPool, InferPool)."""
    import shutil
    shutil.copyfile(OUT_DIR / f"{base}.onnx", OUT_DIR / f"{name}.onnx")
    info = json.loads((OUT_DIR / f"{base}_probe.json").read_text())
    info.update(extra, name=name, pool_impl="pool")
    (OUT_DIR / f"{name}_probe.json").write_text(json.dumps(info, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--pool-impl", choices=("streaming", "swg_pool"), default="streaming",
                    help="swg_pool: InferPool route (depthwise SWG + Pool_hls with PE); writes the '_pool' twin of each case")
    ap.add_argument("--F", type=int, default=F, help="frame cycles (81920 = StreamingMaxPool floor; the Pool route allows less, e.g. 69632)")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for b in a.bits:
        r = model_int_bottleneck(CIN, COUT, b, HW_IN, HW_IN, F=a.F, pool_impl=a.pool_impl)
        verify_with_sim(r)
        base = f"init_cin{CIN}_cout{COUT}_in{HW_IN}_int{b}"
        name = base + ("_pool" if a.pool_impl == "swg_pool" else "")
        if a.pool_impl == "swg_pool":
            _pool_case(base, name, dict(F=a.F, T_out=a.F / (HW_IN // 2) ** 2))
        (OUT_DIR / f"{name}_folding.json").write_text(json.dumps(to_folding_config(r), indent=2))
        v = r.verification
        print(f"{name}: steady {v['steady_cyc_px']:.2f} cyc/px  skip FIFO {r.skip_fifo.depth_words} words  FIFO main {r.params['main_fifo_words']}  "
              f"latency {v['latency_first_out']}  LUT {r.totals['lut']:.0f}  BRAM18 {r.totals['bram18']:.1f}  DSP {r.totals['dsp']:.0f}", flush=True)


if __name__ == "__main__":
    main()
