"""Folding + FIFO config (and the model's prediction) for the downsampling probes. Standard library only.

    python3 hardware/builds/bottleneck_probe_v1/make_dn_folding_configs.py [--bits 4 6 8] [--variants fmpad mvau]

Same case as export_dn_probe.py (16 -> 32, 64x64 input, T_out = 72). Both variants use the "pad_thr" skip order, matching the
exported graph (the residual add's shared input quantizer thresholds the PADDED skip). Writes
inputs/dn_cin16_cout32_in64_int{b}_{variant}_folding.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "MILP" / "analytical"))

from dn_bottleneck import model_dn_bottleneck, to_folding_config, verify_with_sim  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "inputs"
CIN, COUT, V, HW_IN, T_OUT = 16, 32, 4, 64, 72


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
    ap.add_argument("--variants", nargs="+", default=["fmpad", "mvau"], choices=["fmpad", "mvau"])
    ap.add_argument("--pool-impl", choices=("streaming", "swg_pool"), default="streaming",
                    help="swg_pool: InferPool route (depthwise SWG + Pool_hls with PE); writes the '_pool' twin of each case")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for b in a.bits:
        for variant in a.variants:
            r = model_dn_bottleneck(CIN, COUT, V, b, HW_IN, HW_IN, T_out=T_OUT, skip_order="pad_thr", skip_pad=variant, pool_impl=a.pool_impl)
            verify_with_sim(r)
            base = f"dn_cin{CIN}_cout{COUT}_in{HW_IN}_int{b}_{variant}"
            name = base + ("_pool" if a.pool_impl == "swg_pool" else "")
            if a.pool_impl == "swg_pool":
                _pool_case(base, name, {})
            (OUT_DIR / f"{name}_folding.json").write_text(json.dumps(to_folding_config(r), indent=2))
            v = r.verification
            print(f"{name}: steady {v['steady_cyc_px']:.2f} cyc/px  skip FIFO {r.skip_fifo.depth_words} words  latency {v['latency_first_out']}  "
                  f"LUT {r.totals['lut']:.0f}  BRAM18 {r.totals['bram18']:.1f}  DSP {r.totals['dsp']:.0f}", flush=True)


if __name__ == "__main__":
    main()
