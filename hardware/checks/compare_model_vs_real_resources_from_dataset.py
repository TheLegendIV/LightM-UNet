"""Model-vs-real resource calibration check driven by a build's OWN real
per-node folding/quantization choices (PE, SIMD, weight_bits, act_bits, ...),
NOT finn_milp.py's solved layer_bits_folding_*.json.

Why: compare_model_vs_real_resources.py compares the MILP's CHOSEN folding
against what got built. That's only valid if the solved json and the real
build agree node-for-node -- and for S12_dense_nn_upsample_256_v2 they don't:
v2 had a bug in how standalone (join-point) Thresholding_rtl nodes were
bridged from the solve into the deployed folding config, so a chosen-vs-real
comparison there is partly comparing the model against a folding it never
actually solved for. Feeding each real CSV row's OWN recorded PE/SIMD/bits
straight into finn_cost_model.py's cost functions sidesteps that entirely --
every row is scored against itself, independent of solver/build bridging.

Modelled node kinds (cost formula exists in finn_cost_model.py): MVAU, VVAU,
SWU, Thresholding, AddStreams, DuplicateStreams, Concat, Upsample, MaxPool.
Excluded (genuinely unmodelled for LUT/BRAM/DSP -- see finn_cost_model.py):
FIFO, DWC (StreamingDataWidthConverter; the module DOES carry a `dwc_cost()`
formula, deliberately not used here since the user asked to exclude it), and
FMPadding (finn_cost_model.py only prices FMPadding's CYCLES via
`fmpad_cycles`, never its LUT/BRAM/DSP -- there is no resource formula for it
at all).

Known simplifications (this is a TOTAL-only check, not a per-node one):
- MVAU/VVAU rows in this CSV format carry MH/MW (matrix height/width) but not
  the original conv's spatial geometry (H/W/kernel/stride) -- those live only
  on the paired SWU row. So MVAU/VVAU prediction here uses a synthetic 1x1
  LayerGeometry sized so cin*kh*kw/groups == MW and cout == MH (this keeps
  conv_cost_pe_simd's mvu_lut/wm_bram18/mvu_dsp terms exact), called with
  no_activation=False so its internal fused-threshold term stays off (the
  real standalone per-conv threshold is priced separately from its own real
  "Thresholding" CSV row instead -- see below), and its internal SWU
  computation is simply discarded (kh=kw=1 forces `_finn_swu`'s own 1x1
  shortcut to 0,0,0,0). SWU's real resource is priced independently from the
  SWU row's own full real geometry (_finn_swu called directly) and summed in
  separately -- mathematically equivalent to the bundled per_layer call
  finn_milp.py makes, without needing to know which SWU row belongs to which
  MVAU/VVAU row.
- VVAU's real "Channels" (MH) attribute was 1 in both S12 dense 256 builds --
  conv_cost_pe_simd derives `depthwise` from `groups > 1`, which a MH=1
  reconstruction (groups=1) would misclassify, wrongly inflating that one
  node's predicted DSP by a factor of SIMD. Worked around by using groups=2
  (cin=2) as a `depthwise=True` trigger, decoupled from the real MH/channel
  count, which conv_cost_pe_simd never otherwise uses. Only 1 VVAU node
  exists per build, so this affects a small fraction of total DSP either way.
- Every real "Thresholding" row (fused per-conv AND standalone join-point
  alike -- the CSV has no column distinguishing them) is priced independently
  via `threshold_node_cost` using ITS OWN real PE/act_bits. This is exact,
  not an approximation: `_thresholding_rtl_cost`'s formula is identical
  either way, so pricing every real row once, and never re-adding a threshold
  term inside the owning MVAU/VVAU's own prediction (no_activation=False
  above), gives the same total the bundled per-conv path would without
  needing to know which row is which.
- Thresholding rows are priced directly from the CSV's own real `numSteps`
  column (`_THR_RTL_LUT_BASE_PER_PE + _THR_RTL_LUT_PER_NUMSTEP_PE * numSteps`),
  not re-derived from `act_bits` -- for a noActivation=1 threshold, this CSV's
  `act_bits` is `dtype_bits(inputDataType)`, the WIDE pre-threshold
  accumulator type (e.g. INT19), not the threshold's own output precision;
  re-deriving `num_steps = 2**act_bits - 1` from that overshoots by orders of
  magnitude. Thresholding rows also carry no real `ram_style` in this CSV --
  defaults to "block" (LUT is ~ram_style-independent in this formula; only
  the smaller BRAM term would move), gated by `_THR_RTL_BRAM_MIN_CHANNELS_NUMSTEP`
  (real Vivado infers zero BRAM below this channels*numSteps -- see
  finn_cost_model.py).
- MaxPool rows carry PE/NumChannels but not kernel/stride/image-width, so
  only `mp_lut` (= act_bits * channels) and the constant `swu_lut` (=426) are
  priced; `swu_bram18` (needs kernel/stride/width) is left at 0. Only 3
  MaxPool nodes exist per build; real MaxPool BRAM is typically 0 anyway at
  this network's resolution.
- act_bits defaults to 8 where the CSV's own `inputDataType`-derived act_bits
  column is blank (seen for MaxPool rows).

Usage:
    python compare_model_vs_real_resources_from_dataset.py \\
        hardware/builds/12_dense_relu_nearest_conv_upsample_256_v2/results/mvau_lut_calibration_dataset_..._full.csv \\
        hardware/builds/12_dense_relu_nearest_conv_upsample_256_w8_16_v4/results/mvau_lut_calibration_dataset_..._full.csv
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "MILP"))
from finn_cost_model import (  # noqa: E402
    LayerGeometry, calibrated_bram18k, calibrated_lut, conv_cost_pe_simd, stream_node_cost,
    _finn_swu, _THR_RTL_LUT_BASE_PER_PE, _THR_RTL_LUT_PER_NUMSTEP_PE, _THR_RTL_BRAM18_PER_PE_NUMSTEP,
    _THR_RTL_BRAM_MIN_CHANNELS_NUMSTEP,
)

MODELLED_KINDS = {
    "MVAU", "VVAU", "SWU", "Thresholding", "AddStreams", "DuplicateStreams", "Concat", "Upsample", "MaxPool",
}
EXCLUDED_KINDS = {"FIFO", "DWC", "FMPadding"}


def num(v, default=0.0):
    return float(v) if v not in (None, "") else default


def predict_row(r):
    """(lut, bram18_equiv, dsp) predicted for one modelled CSV row."""
    kind = r["node_kind"]

    if kind in ("MVAU", "VVAU"):
        mh, mw = int(num(r["MH"])), int(num(r["MW"]))
        pe, simd = int(num(r["PE"], 1)), int(num(r["SIMD"], 1))
        wbits, abits = num(r["weight_bits"], 8), num(r["act_bits"], 8)
        ram_style = r.get("ram_style") or "auto"
        if kind == "VVAU":
            impl_style = "hls"
            geom = LayerGeometry(
                op_type="Conv2d", name=r["node_name"], stage="",
                cin=2, hin=1, win=1, cout=mh, hout=1, wout=1, kh=mw, kw=1, sh=1, sw=1, groups=2,
            )
        else:
            impl_style = "rtl" if "rtl" in r["op_type"].lower() else "hls"
            geom = LayerGeometry(
                op_type="Conv2d", name=r["node_name"], stage="",
                cin=mw, hin=1, win=1, cout=mh, hout=1, wout=1, kh=1, kw=1, sh=1, sw=1, groups=1,
            )
        cost = conv_cost_pe_simd(
            geom, wbits, abits, pe, simd, ram_style=ram_style,
            force_dsp=True, impl_style=impl_style, no_activation=False,
        )
        lut = calibrated_lut(cost["mvu_lut"], wbits, abits, force_dsp=True)
        bram = calibrated_bram18k(cost["wm_bram18"] + cost["wm_uram18"], wbits, abits, force_dsp=True)
        return lut, bram, cost["mvu_dsp"]

    if kind == "SWU":
        cin = int(num(r["IFMChannels"], 1))
        geom = LayerGeometry(
            op_type="Conv2d", name=r["node_name"], stage="",
            cin=cin, hin=int(num(r["IFMDim_h"], 1)), win=int(num(r["IFMDim_w"], 1)),
            cout=cin, hout=int(num(r["OFMDim_h"], 1)), wout=int(num(r["OFMDim_w"], 1)),
            kh=int(num(r["ConvKernelDim_h"], 1)), kw=int(num(r["ConvKernelDim_w"], 1)),
            sh=int(num(r["Stride_h"], 1)), sw=int(num(r["Stride_w"], 1)),
            dh=int(num(r["Dilation_h"], 1)), dw=int(num(r["Dilation_w"], 1)),
        )
        simd = int(num(r["SIMD"], 1)) or 1
        abits = num(r["act_bits"], 8)
        swu_lut, swu_bram18, swu_uram18, _ = _finn_swu(
            geom, abits, simd, bool(int(num(r["depthwise"]))), bool(int(num(r["parallel_window"]))),
            ram_style=r.get("ram_style") or "distributed",
        )
        lut = calibrated_lut(swu_lut, abits, abits, force_dsp=True)
        bram = calibrated_bram18k(swu_bram18 + swu_uram18, abits, abits, force_dsp=True)
        return lut, bram, 0.0

    if kind == "Thresholding":
        # Use the CSV's own real `numSteps` (FINN's actual node attribute) directly,
        # NOT re-derived from act_bits -- act_bits here is dtype_bits(inputDataType),
        # which for a noActivation=1 threshold is the WIDE pre-threshold accumulator
        # type (e.g. INT19), not the threshold's own output precision; re-deriving
        # num_steps=2**act_bits-1 from that blows up by orders of magnitude. numSteps
        # is the real, ground-truth driver variable (see MILP/calibration.csv's own
        # threshold_lut refit, fit against this exact column).
        pe = int(num(r["PE"], 1)) or 1
        channels = int(num(r["NumChannels"], 1)) or 1
        num_steps = num(r["numSteps"], 255)
        abits_for_calib = 8  # only feeds calibrated_lut/bram's avg_bits table; force_dsp=True makes it a no-op anyway
        raw_lut = pe * (_THR_RTL_LUT_BASE_PER_PE + _THR_RTL_LUT_PER_NUMSTEP_PE * num_steps)
        # Below _THR_RTL_BRAM_MIN_CHANNELS_NUMSTEP, real Vivado infers no dedicated
        # BRAM at all -- see that constant's comment in finn_cost_model.py.
        raw_bram = (
            _THR_RTL_BRAM18_PER_PE_NUMSTEP * pe * num_steps
            if channels * num_steps >= _THR_RTL_BRAM_MIN_CHANNELS_NUMSTEP else 0.0
        )  # ram_style="block" default assumption -- see module docstring
        lut = calibrated_lut(raw_lut, abits_for_calib, abits_for_calib, force_dsp=True)
        bram = calibrated_bram18k(raw_bram, abits_for_calib, abits_for_calib, force_dsp=True)
        return lut, bram, 0.0

    if kind in ("AddStreams", "DuplicateStreams"):
        pe = int(num(r["PE"], 1)) or 1
        cout = int(num(r["NumChannels"], 1)) or 1
        geom = LayerGeometry(
            op_type="Conv2d", name=r["node_name"], stage="",
            cin=cout, hin=1, win=1, cout=cout, hout=1, wout=1, kh=1, kw=1, sh=1, sw=1,
        )
        cost = stream_node_cost("add" if kind == "AddStreams" else "dup", geom, pe)
        return cost["total_lut"], 0.0, 0.0

    if kind in ("Concat", "Upsample"):
        return 0.0, 0.0, 0.0  # FINN v0.10.1 prices both at 0 (PROVISIONAL, see finn_cost_model.py)

    if kind == "MaxPool":
        cin = int(num(r["NumChannels"], 1)) or 1
        abits = num(r["act_bits"], 8)
        return 426 + abits * cin, 0.0, 0.0

    raise ValueError(f"unhandled modelled kind {kind!r}")


def analyze(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    pred = {"lut": 0.0, "bram18_equiv": 0.0, "dsp": 0.0}
    real = {"lut": 0.0, "bram18_equiv": 0.0, "dsp": 0.0}
    n_modelled, n_excluded, unrecognized = 0, 0, {}
    for r in rows:
        kind = r["node_kind"]
        if kind in EXCLUDED_KINDS:
            n_excluded += 1
            continue
        if kind not in MODELLED_KINDS:
            unrecognized[kind] = unrecognized.get(kind, 0) + 1
            continue
        lut, bram, dsp = predict_row(r)
        pred["lut"] += lut
        pred["bram18_equiv"] += bram
        pred["dsp"] += dsp
        real["lut"] += num(r["real_LUT"])
        real["bram18_equiv"] += num(r["real_BRAM18"]) + 2 * num(r["real_BRAM36"])
        real["dsp"] += num(r["real_DSP"])
        n_modelled += 1
    return {
        "n_rows": len(rows), "n_modelled": n_modelled, "n_excluded": n_excluded, "unrecognized": unrecognized,
        "pred": pred, "real": real,
    }


def pct_err(model_v, real_v):
    if real_v == 0:
        return float("nan") if model_v == 0 else float("inf")
    return 100.0 * (model_v - real_v) / real_v


def print_result(label, res):
    p, r = res["pred"], res["real"]
    print(f"=== {label} ({res['n_rows']} total rows, {res['n_modelled']} modelled, "
          f"{res['n_excluded']} excluded FIFO/DWC/FMPadding) ===")
    if res["unrecognized"]:
        print(f"  WARNING: unrecognized node_kind(s) not in MODELLED_KINDS, skipped: {res['unrecognized']}")
    print(f"  LUT:  predicted={p['lut']:10.0f}  real={r['lut']:10.0f}  err={pct_err(p['lut'], r['lut']):+6.1f}%")
    print(f"  BRAM: predicted={p['bram18_equiv']:10.1f}  real={r['bram18_equiv']:10.1f}  "
          f"err={pct_err(p['bram18_equiv'], r['bram18_equiv']):+6.1f}%")
    print(f"  DSP:  predicted={p['dsp']:10.0f}  real={r['dsp']:10.0f}  err={pct_err(p['dsp'], r['dsp']):+6.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_paths", nargs="+", help="build_node_resource_calibration_csv.py per-node CSV(s).")
    args = ap.parse_args()

    results = {p: analyze(p) for p in args.csv_paths}
    for p, res in results.items():
        print_result(p, res)
        print()

    combined = {"pred": {"lut": 0.0, "bram18_equiv": 0.0, "dsp": 0.0}, "real": {"lut": 0.0, "bram18_equiv": 0.0, "dsp": 0.0},
                "n_rows": 0, "n_modelled": 0, "n_excluded": 0, "unrecognized": {}}
    for res in results.values():
        for f in ("lut", "bram18_equiv", "dsp"):
            combined["pred"][f] += res["pred"][f]
            combined["real"][f] += res["real"][f]
        combined["n_rows"] += res["n_rows"]
        combined["n_modelled"] += res["n_modelled"]
        combined["n_excluded"] += res["n_excluded"]
    print_result("COMBINED", combined)


if __name__ == "__main__":
    main()
