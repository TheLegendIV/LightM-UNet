"""Folding + FIFO config (and the model's prediction) for the initial-block probes. Standard library only.

    python3 hardware/builds/bottleneck_probe_v1/make_int_folding_configs.py [--bits 4 6 8]

Same case as export_int_probe.py (1 -> 4 channels, 256x256 -> 128x128, F = 81920). Writes inputs/init_cin1_cout4_in256_int{b}_thrpre_folding.json: the
model now has Thr_m UPSTREAM of the maxpool (landed FINN graph, see int_bottleneck.py), so these are new cases `_thrpre` (they share the ONNX of the
`init_cin1_cout4_in256_int{b}` probes, whose own folding jsons / results are the earlier post-pool model and stay untouched).
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


def _twin(base: str, name: str, extra: dict) -> None:
    """A twin shares the ONNX of its base probe (a different folding / FINN conversion of the same graph)."""
    import shutil
    shutil.copyfile(OUT_DIR / f"{base}.onnx", OUT_DIR / f"{name}.onnx")
    info = json.loads((OUT_DIR / f"{base}_probe.json").read_text())
    info.update(extra, name=name)
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
        name = base + ("_pool" if a.pool_impl == "swg_pool" else "") + "_thrpre"
        _twin(base, name, dict(thr_m_order="pre", **(dict(F=a.F, T_out=a.F / (HW_IN // 2) ** 2, pool_impl="pool") if a.pool_impl == "swg_pool" else {})))
        (OUT_DIR / f"{name}_folding.json").write_text(json.dumps(to_folding_config(r), indent=2))
        v = r.verification
        print(f"{name}: steady {v['steady_cyc_px']:.2f} cyc/px  skip FIFO {r.skip_fifo.depth_words} words  FIFO main {r.params['main_fifo_words']}  "
              f"latency {v['latency_first_out']}  LUT {r.totals['lut']:.0f}  BRAM18 {r.totals['bram18']:.1f}  DSP {r.totals['dsp']:.0f}", flush=True)


if __name__ == "__main__":
    main()
