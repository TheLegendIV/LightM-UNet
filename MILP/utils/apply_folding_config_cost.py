"""Apply a FINN-style folding_config.json (node_name -> {"PE": .., "SIMD": ..},
e.g. FINN's own auto-fold output or a hand-written what-if) to a real,
already-characterized post-fifo-autosize .onnx checkpoint, and price every
compute node with THIS repo's own finn_cost_model.py formulas -- an
independent second LUT/BRAM18K/DSP/cycles/FPS estimate, comparable against
FINN's own `cycles_estimate` node attribute (real ground truth from FINN's
characterization pass, already baked into the checkpoint).

WHY THIS EXISTS: finn_milp.py's cost model is normally driven by this repo's
own Python architecture config (layer_topology.py + the nnU-Net model
definition) plus a SOLVED MILP `per_layer` dict -- there was no way to run
the SAME cost formulas against a folding FINN chose entirely on its own (e.g.
hardware/builds/.../autofold_config_partition2.json, produced by FINN's
`step_target_fps_parallelization` auto-fold heuristic, never by finn_milp.py).
This script closes that gap by reading geometry, datatypes, AND (optionally)
the PE/SIMD folding directly off a real ONNX graph's own node attributes --
no architecture Python module, no torch/nnunetv2 import, no MILP solve
needed. Every compute node in this architecture fully self-describes its own
cost-relevant shape:
  - MVAU_rtl/_hls:                    MW, MH, PE, SIMD, weightDataType, inputDataType
  - ConvolutionInputGenerator_rtl/_hls: ConvKernelDim, Dilation, IFMChannels,
                                        IFMDim, OFMDim, Stride, SIMD, depthwise, parallel_window
  - FMPadding_rtl/_hls:                ImgDim, NumChannels, Padding, SIMD
  - Thresholding_rtl/_hls:             NumChannels, PE, outputDataType, numInputVectors
  - AddStreams_*/DuplicateStreams_*:   NumChannels, PE, numInputVectors
  - StreamingDataWidthConverter_rtl:   inWidth, outWidth
so nothing beyond finn_cost_model.py's EXISTING public functions is needed to
price each one independently -- see `_price_mvau`'s docstring for how it
gets the MVAU's own mvu_lut/mvu_dsp/mvu_cycles out of `conv_cost_pe_simd`
without also pulling in that function's bundled SWU/threshold pricing (this
script prices those as their own separate real nodes instead, since in a
real graph they always already are).

VALIDATED (2026-09-30) with NO folding-config override (i.e. pricing each
node at its own baked-in PE/SIMD) against
hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/
post_fifo_autosize_checkpoints/partition2_baseline_both_off_milpfold_
prefifo_autosize.onnx's own real `cycles_estimate` attributes -- exact
matches, 0 error: MVAU_rtl_10 2,359,296 cycles, ConvolutionInputGenerator_
rtl_3 319,752, Thresholding_rtl_0 / AddStreams_hls_0 / DuplicateStreams_
hls_0 131,072 each. Run with --self-check to reproduce.

Usage (any environment with the `onnx` package -- lightmunet_dev container,
or the FINN container with -e HOME=/tmp/home_dir, or plain host Python):
    # Self-check: does our model reproduce FINN's own measured cycles,
    # using each node's own baked-in PE/SIMD (no folding-config override)?
    python MILP/utils/apply_folding_config_cost.py --onnx <checkpoint.onnx> --self-check

    # Apply a FINN-chosen (or hand-written) folding_config.json's PE/SIMD to
    # the SAME checkpoint's geometry/datatypes, e.g. to get a LUT/BRAM/DSP
    # estimate FINN's own auto-fold never produces on its own:
    python MILP/utils/apply_folding_config_cost.py \\
        --onnx hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/post_fifo_autosize_checkpoints/partition2_baseline_both_off_autofold_prefifo_autosize.onnx \\
        --folding-config hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/autofold_config_partition2.json \\
        --clock-mhz 100
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finn_cost_model import (  # noqa: E402
    IMPL_STYLE_RTL, LayerGeometry, dwc_cost, layer_cost_pe_simd_auto_ram, stream_node_cost, threshold_node_cost,
    _finn_swu,  # a pure per-node pricing primitive, reused directly rather than duplicated -- see _price_swu.
)

# Mirrors finn_milp.py's own XCZU7EV dict (source of truth) -- duplicated
# here rather than imported since finn_milp.py itself pulls in torch/
# nnunetv2/pulp at import time, which this script deliberately avoids.
XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624, "DSP": 1_728, "URAM": 96}


def _dtype_bits(dtype: str) -> int:
    """QONNX DataType string ("UINT8", "INT20", "BIPOLAR", ...) -> bit width."""
    s = dtype.upper()
    if s == "BIPOLAR":
        return 1
    if s.startswith("UINT"):
        return int(s[4:])
    if s.startswith("INT"):
        return int(s[3:])
    raise ValueError(f"Unrecognized QONNX datatype string: {dtype!r}")


def _node_attrs(node: onnx.NodeProto) -> dict:
    out = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT:
            out[a.name] = a.i
        elif a.type == onnx.AttributeProto.STRING:
            out[a.name] = a.s.decode()
        elif a.type == onnx.AttributeProto.INTS:
            out[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.FLOAT:
            out[a.name] = a.f
    return out


def _price_mvau(attrs: dict, pe: int, simd: int, force_dsp: bool) -> dict:
    """The MVAU's own mvu_lut/mvu_dsp/mvu_cycles + weight-memory BRAM/URAM,
    with NO SWU or standalone-threshold cost bundled in (those are priced as
    their own separate real nodes elsewhere in this file, exactly as they
    exist in the real graph).

    Reuses `conv_cost_pe_simd` (via `layer_cost_pe_simd_auto_ram`) UNCHANGED
    through a `LayerGeometry` built so its own real behavior does the
    isolation for us, rather than duplicating its calibrated formula:
      - kh=kw=1, cin=MW, cout=MH: `_finn_swu`'s own first line is
        "a 1x1 kernel gets no SWU node at all" (real hardware fact, not a
        hack), so swu_lut/swu_bram18/swu_cycles come back exactly 0; and
        `max_simd(layer)`/`max_pe(layer)` (cin*kh*kw / cout) evaluate to
        exactly this real MVAU node's own MW/MH.
      - no_activation=False: disables conv_cost_pe_simd's OWN internal
        standalone-threshold estimate (the real noActivation=1 threshold
        that would otherwise be double-counted here, since it's priced as
        its own separate Thresholding_rtl node by `_price_threshold`).
    """
    W = _dtype_bits(attrs["weightDataType"])
    A = _dtype_bits(attrs["inputDataType"])
    MH, MW = attrs["MH"], attrs["MW"]
    hout, wout = attrs["numInputVectors"][-2], attrs["numInputVectors"][-1]
    geom = LayerGeometry(
        op_type="Conv2d", name="", stage="",
        cin=MW, hin=1, win=1, cout=MH, hout=hout, wout=wout,
        kh=1, kw=1, sh=1, sw=1, groups=1, ph=0, pw=0,
    )
    cost = layer_cost_pe_simd_auto_ram(
        geom, W, A, pe, simd, force_dsp=force_dsp, impl_style=IMPL_STYLE_RTL, no_activation=False,
    )
    return {
        "lut": cost["total_lut"], "bram18": cost["wm_bram18"], "uram18": cost["wm_uram18"],
        "dsp": cost["total_dsp"], "cycles": cost["cycles"],
    }


def _price_swu(attrs: dict, simd: int) -> dict:
    kh, kw = attrs["ConvKernelDim"]
    dh, dw = attrs.get("Dilation", [1, 1])
    sh, sw = attrs["Stride"]
    cin = attrs["IFMChannels"]
    hin, win = attrs["IFMDim"]
    hout, wout = attrs["OFMDim"]
    depthwise = bool(attrs.get("depthwise", 0))
    parallel_window = bool(attrs.get("parallel_window", 0))
    act_bits = _dtype_bits(attrs["inputDataType"])
    geom = LayerGeometry(
        op_type="Conv2d", name="", stage="",
        cin=cin, hin=hin, win=win, cout=cin, hout=hout, wout=wout, kh=kh, kw=kw, sh=sh, sw=sw, dh=dh, dw=dw,
    )
    lut, bram18, uram18, cycles = _finn_swu(geom, act_bits, simd, depthwise, parallel_window)
    return {"lut": lut, "bram18": bram18, "uram18": uram18, "dsp": 0, "cycles": cycles}


def _price_fmpadding(attrs: dict, simd: int) -> dict:
    """Cycles only -- FMPadding's own LUT is not modeled in finn_cost_model.py
    at all (a pre-existing, documented gap; see finn_cost_model.md), not
    something specific to this bridge script."""
    hin, win = attrs["ImgDim"]
    cin = attrs["NumChannels"]
    pad = attrs["Padding"]  # [top, left, bottom, right] -- symmetric in this architecture
    ph, pw = pad[0], pad[1]
    cycles = (hin + 2 * ph) * (win + 2 * pw) * math.ceil(cin / simd)
    return {"lut": None, "bram18": 0, "uram18": 0, "dsp": 0, "cycles": cycles}


def _price_threshold(attrs: dict, pe: int) -> dict:
    cout = attrs["NumChannels"]
    act_bits = _dtype_bits(attrs["outputDataType"])
    hout, wout = attrs["numInputVectors"][-2], attrs["numInputVectors"][-1]
    geom = LayerGeometry(op_type="Thresholding", name="", stage="", cin=0, hin=0, win=0,
                          cout=cout, hout=hout, wout=wout, kh=0, kw=0, sh=0, sw=0)
    cost = threshold_node_cost(geom, act_bits, pe)
    return {"lut": cost["total_lut"], "bram18": cost["thr_bram18"], "uram18": cost["thr_uram18"],
            "dsp": 0, "cycles": cost["cycles"]}


def _price_stream(attrs: dict, kind: str, pe: int) -> dict:
    cout = attrs["NumChannels"]
    hout, wout = attrs["numInputVectors"][-2], attrs["numInputVectors"][-1]
    geom = LayerGeometry(op_type="Conv2d", name="", stage="", cin=0, hin=0, win=0,
                          cout=cout, hout=hout, wout=wout, kh=0, kw=0, sh=0, sw=0)
    cost = stream_node_cost(kind, geom, pe)
    return {"lut": cost["total_lut"], "bram18": 0, "uram18": 0, "dsp": 0, "cycles": cost["cycles"]}


def _price_dwc(attrs: dict) -> dict:
    cost = dwc_cost(attrs["inWidth"], attrs["outWidth"])
    # cycles: not modeled anywhere in finn_cost_model.py for this op (see dwc_cost's own docstring).
    return {"lut": cost["total_lut"], "bram18": 0, "uram18": 0, "dsp": 0, "cycles": None}


def price_node(node: onnx.NodeProto, attrs: dict, folding_override: dict, force_dsp: bool):
    """Returns (kind, cost_dict) for a priceable node, or (None, None) to skip
    (StreamingFIFO and any op_type this file doesn't model)."""
    op = node.op_type
    fo = folding_override.get(node.name, {})
    if op.startswith("MVAU"):
        pe, simd = fo.get("PE", attrs["PE"]), fo.get("SIMD", attrs["SIMD"])
        return "mvau", _price_mvau(attrs, pe, simd, force_dsp)
    if op.startswith("ConvolutionInputGenerator"):
        simd = fo.get("SIMD", attrs["SIMD"])
        return "swu", _price_swu(attrs, simd)
    if op.startswith("FMPadding"):
        simd = fo.get("SIMD", attrs["SIMD"])
        return "fmpadding", _price_fmpadding(attrs, simd)
    if op.startswith("Thresholding"):
        pe = fo.get("PE", attrs["PE"])
        return "threshold", _price_threshold(attrs, pe)
    if op.startswith("AddStreams"):
        pe = fo.get("PE", attrs["PE"])
        return "add", _price_stream(attrs, "add", pe)
    if op.startswith("DuplicateStreams"):
        pe = fo.get("PE", attrs["PE"])
        return "dup", _price_stream(attrs, "dup", pe)
    if op.startswith("StreamingDataWidthConverter"):
        return "dwc", _price_dwc(attrs)
    return None, None


def estimate(onnx_path: Path, folding_config_path: Path | None = None, force_dsp: bool = True,
             clock_mhz: float = 100.0) -> dict:
    model = onnx.load(str(onnx_path))
    folding_override = {}
    if folding_config_path is not None:
        with open(folding_config_path) as f:
            raw = json.load(f)
        folding_override = {k: v for k, v in raw.items() if k != "Defaults"}

    rows = []
    for node in model.graph.node:
        attrs = _node_attrs(node)
        kind, cost = price_node(node, attrs, folding_override, force_dsp)
        if kind is None:
            continue
        rows.append({
            "name": node.name, "op_type": node.op_type, "kind": kind,
            "predicted_lut": cost["lut"], "predicted_bram18": cost["bram18"],
            "predicted_uram18": cost.get("uram18", 0), "predicted_dsp": cost["dsp"],
            "predicted_cycles": cost["cycles"], "measured_cycles": attrs.get("cycles_estimate"),
        })

    total_lut = sum(r["predicted_lut"] for r in rows if r["predicted_lut"] is not None)
    total_bram18 = sum(r["predicted_bram18"] for r in rows if r["predicted_bram18"] is not None)
    total_uram18 = sum(r["predicted_uram18"] for r in rows if r["predicted_uram18"] is not None)
    total_dsp = sum(r["predicted_dsp"] for r in rows if r["predicted_dsp"] is not None)
    priced = [r for r in rows if r["predicted_cycles"] is not None]
    bottleneck = max(priced, key=lambda r: r["predicted_cycles"])
    return {
        "rows": rows,
        "total_lut": total_lut, "lut_pct": 100 * total_lut / XCZU7EV["LUT"],
        "total_bram18": total_bram18, "bram_pct": 100 * total_bram18 / XCZU7EV["BRAM_18K"],
        "total_uram18": total_uram18,
        "total_dsp": total_dsp, "dsp_pct": 100 * total_dsp / XCZU7EV["DSP"],
        "bottleneck_name": bottleneck["name"], "bottleneck_op": bottleneck["op_type"],
        "bottleneck_cycles": bottleneck["predicted_cycles"],
        "fps": clock_mhz * 1e6 / bottleneck["predicted_cycles"], "clock_mhz": clock_mhz,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, type=Path, help="A real, characterized .onnx checkpoint "
                    "(supplies every node's geometry/datatypes; PE/SIMD used as the fallback wherever "
                    "--folding-config doesn't override a node).")
    ap.add_argument("--folding-config", type=Path, default=None,
                    help="FINN-style folding_config.json (node_name -> {PE,SIMD}), e.g. an autofold_config_*.json. "
                    "Omit to price the checkpoint's own baked-in PE/SIMD (see --self-check).")
    ap.add_argument("--force-dsp", dest="force_dsp", action="store_true", default=True)
    ap.add_argument("--no-force-dsp", dest="force_dsp", action="store_false")
    ap.add_argument("--clock-mhz", type=float, default=100.0)
    ap.add_argument("--self-check", action="store_true",
                    help="Print predicted-vs-measured cycles per node (measured = the checkpoint's own real "
                    "cycles_estimate attribute) -- only meaningful when NOT overriding folding away from what's "
                    "actually baked into --onnx.")
    args = ap.parse_args()

    result = estimate(args.onnx, args.folding_config, args.force_dsp, args.clock_mhz)

    print(f"{'name':32s} {'op_type':32s} {'pred_LUT':>10s} {'pred_BRAM18':>12s} {'pred_DSP':>9s} "
          f"{'pred_cycles':>12s} {'measured_cycles':>16s}")
    for r in result["rows"]:
        lut_s = f"{r['predicted_lut']:.0f}" if r["predicted_lut"] is not None else "n/a"
        cyc_s = f"{r['predicted_cycles']:.0f}" if r["predicted_cycles"] is not None else "n/a"
        meas_s = str(r["measured_cycles"]) if r["measured_cycles"] is not None else "n/a"
        if args.self_check and r["predicted_cycles"] is not None and r["measured_cycles"]:
            err_pct = 100 * (r["predicted_cycles"] - r["measured_cycles"]) / r["measured_cycles"]
            meas_s += f"  ({err_pct:+.1f}%)"
        print(f"{r['name']:32s} {r['op_type']:32s} {lut_s:>10s} {r['predicted_bram18']:>12.1f} "
              f"{r['predicted_dsp']:>9.0f} {cyc_s:>12s} {meas_s:>16s}")

    print(f"\nTotal predicted LUT:   {result['total_lut']:.0f} ({result['lut_pct']:.2f}% of {XCZU7EV['LUT']} budget)")
    print(f"Total predicted BRAM18K: {result['total_bram18']:.1f} ({result['bram_pct']:.2f}% of {XCZU7EV['BRAM_18K']} budget)")
    print(f"Total predicted URAM18:  {result['total_uram18']:.1f}")
    print(f"Total predicted DSP:   {result['total_dsp']:.0f} ({result['dsp_pct']:.2f}% of {XCZU7EV['DSP']} budget)")
    print(f"Bottleneck: {result['bottleneck_name']} ({result['bottleneck_op']}), "
          f"{result['bottleneck_cycles']:.0f} cycles -> {result['fps']:.2f} FPS @ {result['clock_mhz']:.0f} MHz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
