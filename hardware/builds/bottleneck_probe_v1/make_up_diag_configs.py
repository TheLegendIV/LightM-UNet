"""Diagnostic twins of the INT4 `noconv` up4 probe (32 -> 16, 32x32 -> 64x64, T_out = 18). Standard library only.

    python3 hardware/builds/bottleneck_probe_v1/make_up_diag_configs.py

Why: `up_cin32_cout16_in32_int4_noconv` landed every PE/SIMD as predicted but measured 20.5 cyc/px against the 18 target (model: 17.6). Two groups
of suspects, one probe each (all share the ONNX of the baseline; only the folding json and the `_probe.json` differ):

  PE route (the join chain runs at 16 cyc/px = 89% of F with PE 1; a DWC / Thresholding / Add with any per-word overhead has no slack):
    _pe2    join_pe = 2: Thr_s, Add, Thr_out at PE 2  (UpNN->Thr_s DWC 64 -> 8 bit, 8 words per pixel)
    _pe4    join_pe = 4: Thr_s, Add, Thr_out at PE 4  (64 -> 16 bit, 4 words per pixel)
  FIFO route (our forced depths vs FINN's stock sizing, see FINN_AGENT_HANDOFF.md section 16):
    _fjoin  ext-side join FIFO (Thr_e -> Add, the `is_skip` edge) 17 -> 512 words  (FINN stock: 484)
    _fupnn  FIFO in front of UpsampleNearestNeighbour (dwc -> upnn) 8 -> 40 words (one input row of 32 pixels, so the 1x1 projection can run ahead
            while the upsampler re-emits the row), and upnn -> dwc 8 -> 91 words (FINN stock)
    _fall   both FIFO changes together
  Ceiling:
    _pe4fall  join_pe = 4 plus both FIFO changes

Reading the result is in the handoff. Writes inputs/up_cin32_cout16_in32_int4_noconv_<tag>{.onnx,_probe.json,_folding.json}.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "MILP" / "analytical"))

from up_bottleneck import model_up_bottleneck, to_folding_config, verify_with_sim  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "inputs"
CIN, COUT, V, HW_IN, T_OUT, BITS = 32, 16, 4, 32, 18, 4
BASE = f"up_cin{CIN}_cout{COUT}_in{HW_IN}_int{BITS}_noconv"

JOIN_FIFO = 512        # FINN stock sizing of Thr_e -> Add was 484
UPNN_IN_FIFO = 40      # one input row (32 pixels) of 64-bit words plus margin
UPNN_OUT_FIFO = 91     # FINN stock sizing of UpNN -> DWC

# tag -> (join_pe, FIFO groups)
VARIANTS = {
    "pe2": (2, ()), "pe4": (4, ()),
    "fjoin": (1, ("join",)), "fupnn": (1, ("upnn",)), "fall": (1, ("join", "upnn")),
    "pe4fall": (4, ("join", "upnn")),
}


def _set_depth(cfg: dict, match, depth: int, what: str) -> dict:
    hits = [f for f in cfg["fifos"] if match(f)]
    if not hits:
        raise RuntimeError(f"{what}: no FIFO edge matches in the folding config")
    change = []
    for f in hits:
        change.append(dict(edge=[f["producer"], f["consumer"]], was=f["depth"], now=max(f["depth"], depth)))
        f["depth"] = max(f["depth"], depth)
    return dict(what=what, changes=change)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tag, (join_pe, groups) in VARIANTS.items():
        r = model_up_bottleneck(CIN, COUT, V, BITS, HW_IN, HW_IN, T_out=T_OUT, skip_conv=False, join_pe=join_pe)
        verify_with_sim(r)
        cfg = to_folding_config(r)
        overrides = []
        if "join" in groups:
            overrides.append(_set_depth(cfg, lambda f: f["is_skip"], JOIN_FIFO, "ext-side join FIFO (Thr_e -> Add)"))
            cfg["predicted"]["skip_fifo_words"] = JOIN_FIFO
        if "upnn" in groups:
            overrides.append(_set_depth(cfg, lambda f: (f["producer"], f["consumer"]) == ("dwc", "upnn"), UPNN_IN_FIFO, "FIFO in front of UpNN"))
            overrides.append(_set_depth(cfg, lambda f: (f["producer"], f["consumer"]) == ("upnn", "dwc"), UPNN_OUT_FIFO, "FIFO after UpNN"))
        cfg["diag"] = dict(tag=tag, base=BASE, join_pe=join_pe, overrides=overrides)
        name = f"{BASE}_{tag}"
        shutil.copyfile(OUT_DIR / f"{BASE}.onnx", OUT_DIR / f"{name}.onnx")
        info = json.loads((OUT_DIR / f"{BASE}_probe.json").read_text())
        info.update(name=name, diag=tag, join_pe=join_pe)
        (OUT_DIR / f"{name}_probe.json").write_text(json.dumps(info, indent=2))
        (OUT_DIR / f"{name}_folding.json").write_text(json.dumps(cfg, indent=2))
        v = r.verification
        pes = {x.name: x.pe for x in r.nodes if x.name in ("Thr_s", "Add", "Thr_out")}
        print(f"{name}: join PE {pes}  sim steady {v['steady_cyc_px']:.2f} cyc/px  LUT {r.totals['lut']:.0f}  "
              f"overrides {[o['what'] for o in overrides]}", flush=True)


if __name__ == "__main__":
    main()
