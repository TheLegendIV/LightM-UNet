"""Folding + FIFO config (and the model's prediction) for the initial-block probes. Standard library only.

    python3 hardware/builds/bottleneck_probe_v1/make_int_folding_configs.py [--bits 4 6 8]      # StreamingMaxPool route, F = 81920
    python3 hardware/builds/bottleneck_probe_v1/make_int_folding_configs.py --pool-set           # the 3 InferPool-route probes, INT4

Same case as export_int_probe.py (1 -> 4 channels, 256x256 -> 128x128). The model has Thr_m UPSTREAM of the maxpool (the landed FINN graph, see
int_bottleneck.py), so every case is a `_thrpre` twin: it shares the ONNX of `init_cin1_cout4_in256_int{b}`, whose own folding jsons / results are the
earlier post-pool model and stay untouched.

StreamingMaxPool route (`init_..._int{b}_thrpre`): measured 13.04 cyc/px against 5.0 (StreamingMaxPool_hls at one channel looks ~3.3 cycles per input pixel).
InferPool route (`--pool-set`, INT4): depthwise SWG + Pool_hls, three variants of the same ONNX:
  init_..._int4_pool_thrpre      SWG_p parallel_window 0, F = 81920 (5.0 cyc/px): the first Pool probe measured ~82k cycles per frame at F = 69632
  init_..._int4_poolpw_thrpre    SWG_p parallel_window 1 (one 2x2 window per word, DWC to Pool_hls), F = 69632 (4.25 cyc/px)
  init_..._int4_poolpw81_thrpre  SWG_p parallel_window 1, F = 81920   (separates the budget from the window mode)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "MILP" / "analytical"))

from int_bottleneck import model_int_bottleneck, to_folding_config, verify_with_sim  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "inputs"
CIN, COUT, HW_IN, F_STREAMING = 1, 4, 256, 81920
POOL_SET = (("pool", False, 81920), ("poolpw", True, 69632), ("poolpw81", True, 81920))   # (tag, parallel_window, F)


def _twin(base: str, name: str, extra: dict) -> None:
    """A twin shares the ONNX of its base probe (a different folding / FINN conversion of the same graph)."""
    shutil.copyfile(OUT_DIR / f"{base}.onnx", OUT_DIR / f"{name}.onnx")
    info = json.loads((OUT_DIR / f"{base}_probe.json").read_text())
    info.update(extra, name=name, thr_m_order="pre")
    (OUT_DIR / f"{name}_probe.json").write_text(json.dumps(info, indent=2))


def _write(base: str, name: str, r, extra: dict) -> None:
    verify_with_sim(r)
    _twin(base, name, extra)
    (OUT_DIR / f"{name}_folding.json").write_text(json.dumps(to_folding_config(r), indent=2))
    v, n = r.verification, {x.name: x for x in r.nodes}
    pool = f"  SWG_p {n['SWG_p'].frame_cycles} Pool {n['Pool'].frame_cycles} cycles/frame" if "Pool" in n else ""
    print(f"{name}: F {r.params['F']} (T {r.params['T']:.2f})  sim steady {v['steady_cyc_px']:.2f} cyc/px  skip FIFO {r.skip_fifo.depth_words} words  "
          f"latency {v['latency_first_out']}  LUT {r.totals['lut']:.0f}  BRAM18 {r.totals['bram18']:.1f}  DSP {r.totals['dsp']:.0f}{pool}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--pool-set", action="store_true", help="write the 3 InferPool-route probes (INT4 only) instead of the StreamingMaxPool ones")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.pool_set:
        base = f"init_cin{CIN}_cout{COUT}_in{HW_IN}_int4"
        for tag, pw, F in POOL_SET:
            r = model_int_bottleneck(CIN, COUT, 4, HW_IN, HW_IN, F=F, pool_impl="swg_pool", pool_pw=pw)
            _write(base, f"{base}_{tag}_thrpre", r, dict(F=F, T_out=F / (HW_IN // 2) ** 2, pool_impl="pool", pool_pw=pw))
        return
    for b in a.bits:
        base = f"init_cin{CIN}_cout{COUT}_in{HW_IN}_int{b}"
        _write(base, f"{base}_thrpre", model_int_bottleneck(CIN, COUT, b, HW_IN, HW_IN, F=F_STREAMING), {})


if __name__ == "__main__":
    main()
