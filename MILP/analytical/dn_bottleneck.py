"""Analytical model of one ENet DOWNSAMPLING bottleneck on FINN (companion of reg_bottleneck.py / bottleneck.py).

    main:  Dup -> SWG_r -> MVAU_r (2x2, stride 2) -> Thr_r -> FMPad -> SWG_m -> MVAU_m (3x3) -> Thr_m -> MVAU_e (1x1) -> Thr_e -+
    skip:  Dup -> MaxPool -> Thr_s -> [skip FIFO] -> FMPad_c (channel pad Cin->Cout) ------------------------------------------+-> Add -> Thr_out

Differences to the regular bottleneck (see analytical.md "Downsampling bottleneck"):
* Two pixel domains: Dup, MaxPool and the strided window generator see the H x W input pixels, everything after the stride sees
  (H/2) x (W/2). Balancing is therefore on FRAME cycles: every node must satisfy  pixels_node * cyc_per_pixel <= F.
  Reported cyc_px is always cycles per OUTPUT pixel (= frame cycles / (H/2*W/2)), so util = cyc_px / T_out = frame cycles / F.
* MaxPool is not foldable (all channels per cycle, ~1.25*H*W cycles): F below that is rejected.
* The channel pad Cin -> Cout of the skip is an FMPadding on a regrouped stream (rows = pixels, columns = C/s, s channels per
  column), knob pad_group = s (default: the largest common divisor of Cin and Cout-Cin). skip_pad="mvau" models the original
  padded-identity 1x1 MVAU (INT8 weights) instead, for comparison.
* The deep FIFO sits on the skip branch (the short-latency branch, as in the regular block); main-branch FIFOs stay tiny except
  the sizes the simulation proves necessary.

Run: python3 dn_bottleneck.py --cin 16 --cout 32 --v 4 --bits 4 --height 64 --width 64 --T-out 72 [--verify]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bottleneck as reg  # noqa: E402
from bottleneck import (  # noqa: E402
    BottleneckResult, DwcResult, NodeResult, SkipFifo, _dwc, _fifo_attrs, _fifo_bram18, _geom, _min_pe, _retarget_threshold,
    _search_mvau, finalize_fifo_costs, search_swg_pool,
)

fcm = reg.fcm
SKIP_ORDERS = ("thr_pad", "pad_thr")
SKIP_PADS = ("fmpad", "mvau")


def pad_group_candidates(cin: int, cout: int) -> list[int]:
    return fcm.divisors(math.gcd(cin, cout - cin))


def model_dn_bottleneck(
    cin: int, cout: int, v: int, bits: int, height: int, width: int, F: int | None = None, T_out: float | None = None,
    pad_group: int | None = None, skip_order: str = "thr_pad", skip_pad: str = "fmpad", pool_impl: str = "streaming",
) -> BottleneckResult:
    """cin -> cout (cout > cin) downsampling block, Cmid = cout/v, input map height x width (even), uniform INT `bits`.
    Budget: F frame cycles, or T_out cycles per output pixel (F = T_out * (height/2 * width/2))."""
    if height % 2 or width % 2:
        raise ValueError("height and width must be even (2x2 stride-2 pooling and reduce)")
    if cout <= cin:
        raise ValueError("downsampling block pads channels: need cout > cin")
    if cout % v:
        raise ValueError(f"Cout={cout} not divisible by v={v}")
    if pool_impl not in ("streaming", "swg_pool"):
        raise ValueError("pool_impl in (streaming = StreamingMaxPool, swg_pool = depthwise SWG + Pool_hls with PE)")
    if skip_order not in SKIP_ORDERS or skip_pad not in SKIP_PADS:
        raise ValueError(f"skip_order in {SKIP_ORDERS}, skip_pad in {SKIP_PADS}")
    if (F is None) == (T_out is None):
        raise ValueError("give exactly one of F (frame cycles) or T_out (cycles per output pixel)")
    ho, wo = height // 2, width // 2
    px_in, px_out = height * width, ho * wo
    F = int(F if F is not None else math.floor(T_out * px_out))
    T = F / px_out
    cmid = cout // v
    A = bits
    s = None
    if skip_pad == "fmpad":
        cands = pad_group_candidates(cin, cout)
        s = pad_group if pad_group is not None else max(cands)
        if s not in cands:
            raise ValueError(f"pad_group={s} must divide both Cin={cin} and Cout-Cin={cout - cin}; options {cands}")
        if (cout // s) * px_out > F:
            raise ValueError(f"channel pad needs {cout // s} cycles/pixel > T_out={T:.1f}; raise pad_group (options {cands})")
    res = BottleneckResult(params=dict(
        cin=cin, cmid=cmid, cout=cout, v=v, z=v, T=T, F=F, bits=bits, k=3, dilation=1, stride=2, height=height, width=width,
        hout=ho, wout=wo, pad=1, pad_group=s, skip_order=skip_order, skip_pad=skip_pad, T_in=F / px_in, block="down", pool_impl=pool_impl,
    ))

    # ---- geometries
    g_r = _geom("reduce", cin, cmid, height, width, ho, wo, k=2, s=2, d=1, p=0)
    g_m = _geom("mid", cmid, cmid, ho, wo, ho, wo, k=3, s=1, d=1, p=1)
    g_e = _geom("expand", cmid, cout, ho, wo, ho, wo)
    g_mp = _geom("pool", cin, cin, height, width, ho, wo, k=2, s=2, op="MaxPool2d")
    g_dup = _geom("dup", cin, cin, height, width, height, width, op="Dup")
    g_join = _geom("join", cout, cout, ho, wo, ho, wo, op="Thresholding")
    g_tsc = _geom("ts_cin", cin, cin, ho, wo, ho, wo, op="Thresholding")

    # ---- main MVAUs: largest cycles <= F
    pe_r, simd_r, c_r = _search_mvau(g_r, bits, F)
    pe_m, simd_m, c_m = _search_mvau(g_m, bits, F)
    pe_e, simd_e, c_e = _search_mvau(g_e, bits, F)
    tpe_r = _min_pe(cmid, px_out, F, "Thr_r")
    tpe_m = _min_pe(cmid, px_out, F, "Thr_m")
    tpe_e = _min_pe(cout, px_out, F, "Thr_e")
    c_r = _retarget_threshold(c_r, g_r, A, tpe_r)
    c_m = _retarget_threshold(c_m, g_m, A, tpe_m)
    c_e = _retarget_threshold(c_e, g_e, A, tpe_e)

    # ---- maxpool (not foldable): floor on F
    if pool_impl == "streaming":
        c_mp = fcm.maxpool_cost(g_mp, A)
        if c_mp["cycles"] > F:
            raise ValueError(f"F={F} is below the maxpool floor of {c_mp['cycles']} cycles/frame (not foldable)")
        pe_mp_in, pool_first, pool_last = cin, "MaxPool", "MaxPool"      # StreamingMaxPool takes all channels per cycle
    else:
        sp = search_swg_pool(g_mp, A, F)
        pe_mp_in, pool_first, pool_last = sp["pe"], "SWG_p", "Pool"      # Pool folded over channels: narrowest PE that fits F

    # ---- stream nodes
    pe_d = _min_pe(cin, px_in, F, "Dup")
    pe_a = _min_pe(cout, px_out, F, "Add")
    pe_to = _min_pe(cout, px_out, F, "Thr_out")
    c_dup = fcm.stream_node_cost("dup", g_dup, pe_d)
    c_add = fcm.stream_node_cost("add", g_join, pe_a)
    c_to = fcm.threshold_node_cost(g_join, A, pe_to)

    # ---- skip branch
    mvau_s = None
    if skip_pad == "mvau":
        g_s = _geom("skip_mvau", cin, cout, ho, wo, ho, wo)
        pe_s, simd_s, c_s = _search_mvau(g_s, bits, F, weight_bits=8)
        tpe_s = _min_pe(cout, px_out, F, "Thr_s")
        c_s = _retarget_threshold(c_s, g_s, A, tpe_s)
        pe_ts, c_ts, ts_ch = tpe_s, None, cout
        mvau_s = (pe_s, simd_s, c_s)
    else:
        ts_ch = cin if skip_order == "thr_pad" else cout
        pe_ts = _min_pe(ts_ch, px_out, F, "Thr_s")
        c_ts = fcm.threshold_node_cost(g_tsc if ts_ch == cin else g_join, A, pe_ts)

    acc_r, acc_m, acc_e = c_r["acc_bits"], c_m["acc_bits"], c_e["acc_bits"]
    add_bits = A + 1
    simd_swu_r, simd_swu_m = c_r["simd_swu"], c_m["simd_swu"]

    def row(name, op, pe, simd, frame_cycles, lut=0.0, bram=0.0, uram=0.0, dsp=0.0, in_w=0, out_w=0):
        return NodeResult(name=name, op=op, pe=pe, simd=simd, cyc_px=frame_cycles / px_out, frame_cycles=int(frame_cycles),
                          lut=lut, bram18=bram, uram=uram, dsp=dsp, in_width_bits=in_w, out_width_bits=out_w)

    nodes = res.nodes
    nodes.append(row("Dup", "DuplicateStreams", pe_d, 0, c_dup["cycles"], c_dup["total_lut"], in_w=pe_d * A, out_w=pe_d * A))
    nodes.append(row("SWG_r", "ConvolutionInputGenerator_rtl 2x2 s2", 0, simd_swu_r, c_r["swu_cycles"], c_r["swu_lut"],
                     c_r["swu_bram18"], c_r["swu_uram18"], in_w=simd_swu_r * A, out_w=simd_swu_r * A))
    nodes.append(row("MVAU_r", "MVAU rtl 2x2 s2", pe_r, simd_r, c_r["mvu_cycles"], c_r["mvu_lut"], c_r["wm_bram18"], c_r["wm_uram18"],
                     c_r["mvu_dsp"], in_w=simd_r * A, out_w=pe_r * acc_r))
    nodes.append(row("Thr_r", "Thresholding_rtl", tpe_r, 0, px_out * (cmid // tpe_r), c_r["thr_lut"], c_r["thr_bram18"],
                     in_w=tpe_r * acc_r, out_w=tpe_r * A))
    nodes.append(row("FMPad", "FMPadding 3x3", 0, simd_swu_m, c_m["fmpad_cycles"], in_w=simd_swu_m * A, out_w=simd_swu_m * A))
    nodes.append(row("SWG_m", "ConvolutionInputGenerator_rtl", 0, simd_swu_m, c_m["swu_cycles"], c_m["swu_lut"], c_m["swu_bram18"],
                     c_m["swu_uram18"], in_w=simd_swu_m * A, out_w=simd_swu_m * A))
    nodes.append(row("MVAU_m", "MVAU rtl 3x3", pe_m, simd_m, c_m["mvu_cycles"], c_m["mvu_lut"], c_m["wm_bram18"], c_m["wm_uram18"],
                     c_m["mvu_dsp"], in_w=simd_m * A, out_w=pe_m * acc_m))
    nodes.append(row("Thr_m", "Thresholding_rtl", tpe_m, 0, px_out * (cmid // tpe_m), c_m["thr_lut"], c_m["thr_bram18"],
                     in_w=tpe_m * acc_m, out_w=tpe_m * A))
    nodes.append(row("MVAU_e", "MVAU rtl 1x1", pe_e, simd_e, c_e["mvu_cycles"], c_e["mvu_lut"], c_e["wm_bram18"], c_e["wm_uram18"],
                     c_e["mvu_dsp"], in_w=simd_e * A, out_w=pe_e * acc_e))
    nodes.append(row("Thr_e", "Thresholding_rtl", tpe_e, 0, px_out * (cout // tpe_e), c_e["thr_lut"], c_e["thr_bram18"],
                     in_w=tpe_e * acc_e, out_w=tpe_e * A))
    if pool_impl == "streaming":
        nodes.append(row("MaxPool", "StreamingMaxPool_hls", 0, 0, c_mp["cycles"], c_mp["total_lut"], c_mp["swu_bram18"],
                         in_w=cin * A, out_w=cin * A))
    else:
        nodes.append(row("SWG_p", "ConvolutionInputGenerator depthwise 2x2 s2", 0, sp["pe"], sp["swg_cycles"], sp["swg_lut"], sp["swg_bram18"],
                         sp["swg_uram18"], in_w=sp["pe"] * A, out_w=sp["pe"] * A))
        nodes.append(row("Pool", "Pool_hls", sp["pe"], 0, sp["pool_cycles"], sp["pool_lut"], in_w=sp["pe"] * A, out_w=sp["pe"] * A))
    if skip_pad == "mvau":
        pe_s, simd_s, c_s = mvau_s
        acc_s = c_s["acc_bits"]
        nodes.append(row("MVAU_s", "MVAU rtl 1x1 identity (INT8 w)", pe_s, simd_s, c_s["mvu_cycles"], c_s["mvu_lut"], c_s["wm_bram18"],
                         c_s["wm_uram18"], c_s["mvu_dsp"], in_w=simd_s * A, out_w=pe_s * acc_s))
        nodes.append(row("Thr_s", "Thresholding_rtl (skip)", pe_ts, 0, px_out * (cout // pe_ts), c_s["thr_lut"], c_s["thr_bram18"],
                         in_w=pe_ts * acc_s, out_w=pe_ts * A))
    else:
        nodes.append(row("Thr_s", "Thresholding_rtl (skip)", pe_ts, 0, c_ts["cycles"], c_ts["total_lut"], c_ts["thr_bram18"],
                         in_w=pe_ts * A, out_w=pe_ts * A))
        nodes.append(row("FMPad_c", "FMPadding channel pad", 0, s, px_out * (cout // s), in_w=s * A, out_w=s * A))
    nodes.append(row("Add", "AddStreams", pe_a, 0, c_add["cycles"], c_add["total_lut"], in_w=pe_a * A, out_w=pe_a * add_bits))
    nodes.append(row("Thr_out", "Thresholding_rtl (join)", pe_to, 0, c_to["cycles"], c_to["total_lut"], c_to["thr_bram18"],
                     in_w=pe_to * add_bits, out_w=pe_to * A))

    # ---- DWCs (every edge with a width mismatch)
    dw = res.dwcs
    _dwc("Dup->SWG_r", pe_d * A, simd_swu_r * A, cin, pe_d, simd_swu_r, px_in, px_out, dw)
    _dwc("MVAU_r->Thr_r", pe_r * acc_r, tpe_r * acc_r, cmid, pe_r, tpe_r, px_out, px_out, dw)
    _dwc("Thr_r->FMPad", tpe_r * A, simd_swu_m * A, cmid, tpe_r, simd_swu_m, px_out, px_out, dw)
    _dwc("MVAU_m->Thr_m", pe_m * acc_m, tpe_m * acc_m, cmid, pe_m, tpe_m, px_out, px_out, dw)
    _dwc("Thr_m->MVAU_e", tpe_m * A, simd_e * A, cmid, tpe_m, simd_e, px_out, px_out, dw)
    _dwc("MVAU_e->Thr_e", pe_e * acc_e, tpe_e * acc_e, cout, pe_e, tpe_e, px_out, px_out, dw)
    _dwc("Thr_e->Add", tpe_e * A, pe_a * A, cout, tpe_e, pe_a, px_out, px_out, dw)
    _dwc(f"Dup->{pool_first}", pe_d * A, pe_mp_in * A, cin, pe_d, pe_mp_in, px_in, px_out, dw)
    if skip_pad == "mvau":
        _dwc(f"{pool_last}->MVAU_s", pe_mp_in * A, simd_s * A, cin, pe_mp_in, simd_s, px_out, px_out, dw)
        _dwc("MVAU_s->Thr_s", pe_s * acc_s, pe_ts * acc_s, cout, pe_s, pe_ts, px_out, px_out, dw)
        _dwc("FIFO->Add (skip)", pe_ts * A, pe_a * A, cout, pe_ts, pe_a, px_out, px_out, dw)
    elif skip_order == "thr_pad":
        _dwc(f"{pool_last}->Thr_s", pe_mp_in * A, pe_ts * A, cin, pe_mp_in, pe_ts, px_out, px_out, dw)
        _dwc("FIFO->FMPad_c", pe_ts * A, s * A, cin, pe_ts, s, px_out, px_out, dw)   # widen AFTER the (narrow) FIFO
        _dwc("FMPad_c->Add", s * A, pe_a * A, cout, s, pe_a, px_out, px_out, dw)
    else:
        _dwc(f"{pool_last}->FMPad_c", pe_mp_in * A, s * A, cin, pe_mp_in, s, px_out, px_out, dw)
        _dwc("FMPad_c->Thr_s", s * A, pe_ts * A, cout, s, pe_ts, px_out, px_out, dw)
        _dwc("FIFO->Add (skip)", pe_ts * A, pe_a * A, cout, pe_ts, pe_a, px_out, px_out, dw)
    _dwc("Add->Thr_out", pe_a * add_bits, pe_to * add_bits, cout, pe_a, pe_to, px_out, px_out, dw)
    for d in dw:
        if d.cyc_px > T + 1e-9:
            res.warnings.append(f"DWC {d.edge} needs {d.cyc_px:.1f} cyc/px (output pixels) > T_out={T:.1f}")

    # ---- latency + skip FIFO estimate (first pixel through; input pixels arrive every T_in = F / (H*W) cycles)
    T_in = F / px_in
    cyc = {x.name: x.cyc_px for x in nodes}
    thr_px = lambda ch, pe: ch / pe
    dup_px = cin / pe_d
    t_swg_r = dup_px + (width + 2) * T_in                       # first 2x2 window complete (pixel (1,1) is index W+1)
    t_first_r = t_swg_r + cyc["MVAU_r"] + thr_px(cmid, tpe_r)
    cf_m = cmid / simd_swu_m
    n_fill = wo + 2                                              # real pixels the 3x3 window generator needs (pad 1)
    pad_top = (wo + 2) * cf_m
    t_window = max(t_first_r, pad_top) + (n_fill - 1) * T
    swg_px = 9 * cmid / simd_m
    l_main = t_window + swg_px + cyc["MVAU_m"] + thr_px(cmid, tpe_m) + cyc["MVAU_e"] + thr_px(cout, tpe_e)
    skip_ch = ts_ch
    t_skip = dup_px + (width + 2) * T_in + thr_px(skip_ch, pe_ts) + (cout // s if s else 0)
    if skip_pad == "mvau":
        t_skip += cyc["MVAU_s"]
    words_px = skip_ch // pe_ts
    skip_px = max(1, math.ceil((l_main - t_skip) / T)) + 1
    depth = skip_px * words_px
    fifo_w = pe_ts * A
    res.skip_fifo = SkipFifo(
        width_bits=fifo_w, depth_words=depth, bits=fifo_w * depth, bram18_if_block=_fifo_bram18(fifo_w, depth),
        lutram_luts_if_distributed=math.ceil(fifo_w * depth / 64), pixels_buffered=skip_px, pe=pe_ts,
    )
    res.latency_first_out_cycles = int(math.ceil(l_main + cout / pe_a + cout / pe_to))
    res.frame_cycles = int(res.latency_first_out_cycles + (px_out - 1) * T)

    dwc_lut = sum(d.lut for d in dw)
    res.totals = dict(
        lut=sum(x.lut for x in nodes) + dwc_lut, bram18=sum(x.bram18 for x in nodes) + res.skip_fifo.bram18_if_block,
        uram=sum(x.uram for x in nodes), dsp=sum(x.dsp for x in nodes), dwc_lut=dwc_lut,
    )
    mv = [x for x in nodes if x.op.startswith("MVAU")]
    cycs = [x.cyc_px for x in mv]
    res.balance = dict(max_cyc_px=max(cycs), min_cyc_px=min(cycs), min_over_max=min(cycs) / max(cycs),
                       mean_util=sum(cycs) / len(cycs) / T, scope="MVAU nodes")
    if max(x.frame_cycles for x in nodes) > F:
        res.warnings.append(f"slowest node {max(x.frame_cycles for x in nodes)} cycles/frame exceeds F={F}")
    for x in mv:
        if x.frame_cycles < F * 0.999:
            res.warnings.append(f"{x.name} runs at {x.frame_cycles / F:.0%} of F: no (PE, SIMD) divisor pair lands exactly on F")
    if skip_pad == "fmpad":
        res.warnings.append("FMPad_c / FMPad: FMPadding LUT is not calibrated in finn_cost_model (priced 0); channel pad assumes the "
                            "regrouped-stream FMPadding works in FINN (untested)")
    return res


# escalation schedule: (elastic scale, FIFO after FMPad holds the next frame's padding, uniform depth multiplier)
_SIZING_SCHEDULE = ((1, False, 1), (1, True, 1), (2, True, 1), (2, True, 2), (4, True, 2), (4, True, 4))


def verify_with_sim(r: BottleneckResult, fifo_depth: int = 2, tol: float = 0.02, max_tries: int = 8, shrink: bool = True,
                    tol_soft: float = 0.03, fifo_mem: str = "auto") -> dict:
    """Same recipe as bottleneck.verify_with_sim, on the downsampling network: (1) paced input at T_in with an unbounded skip
    FIFO measures the skip depth really needed, (2) a saturated run doubles one uniform depth (and the elastic FIFOs in front of
    SWG_r / FMPad) until the steady cyc/px per OUTPUT pixel reaches T_out, then every FIFO shrinks to its observed occupancy."""
    from dn_bottleneck_sim import UNBOUNDED, simulate_dn

    p = r.params
    T = p["T"]
    cf_m = p["cmid"] // next(x for x in r.nodes if x.name == "SWG_m").simd
    cf_r = p["cin"] // next(x for x in r.nodes if x.name == "SWG_r").simd
    words_px = (p["cin"] if (p["skip_pad"] == "fmpad" and p["skip_order"] == "thr_pad") else p["cout"]) // r.skip_fifo.pe
    # The sliding windows serve ONE frame at a time: the FIFO in front of each must hold the next frame's first window
    # (3x3: pad*Wo + pad + 1 pixels = Wo + 2; strided 2x2: W + 2 input pixels), see bottleneck.verify_with_sim.
    emap = {"FMPad": (p["wout"] + 2) * cf_m + 2, "SWG_r": (p["width"] + 2) * cf_r + 2}
    elastic = emap["FMPad"]
    pad_out = (p["wout"] + 2) * cf_m + 2
    paced = simulate_dn(r, inject_interval=p["T_in"], skip_depth=UNBOUNDED, fifo_depth=fifo_depth, elastic_map=emap, frames=2)
    need = paced.fifo_max["skip FIFO"]
    if need > r.skip_fifo.depth_words:
        d = need + words_px
        r.warnings.append(f"skip FIFO raised from {r.skip_fifo.depth_words} to {d} words by simulation")
        f = r.skip_fifo
        f.depth_words, f.bits, f.pixels_buffered = d, f.width_bits * d, math.ceil(d / words_px)
        f.bram18_if_block, f.lutram_luts_if_distributed = _fifo_bram18(f.width_bits, d), math.ceil(f.width_bits * d / 64)
    sat = None
    tries = 0
    emap0 = dict(emap)
    for tries, (es, with_pad, um) in enumerate(_SIZING_SCHEDULE[:max_tries], 1):
        depth = fifo_depth * um
        emap = {k: max(v * es, depth) for k, v in emap0.items()}
        elastic = emap["FMPad"]
        extra = {"FMPad->out": pad_out * es} if with_pad else {}
        sat = simulate_dn(r, inject_interval=0, skip_depth=r.skip_fifo.depth_words, fifo_depth=depth, elastic_map=emap,
                          fifo_depths=extra, frames=3)
        if not sat.deadlock and sat.steady_cyc_px <= T * (1 + tol):
            break
    ok = (not sat.deadlock) and sat.steady_cyc_px <= T * (1 + tol)
    if not ok and not sat.deadlock and sat.steady_cyc_px <= T * (1 + tol_soft):
        # inherent per-frame gap: the sliding window needs the next frame's first window (padding rows are read at one word
        # per cycle) before it can emit, which no FIFO hides completely; accept up to tol_soft but say so
        r.warnings.append(f"steady {sat.steady_cyc_px:.2f} cyc/px is {sat.steady_cyc_px / T - 1:.1%} above T: residual per-frame "
                          f"window-fill gap that FIFO sizing cannot remove")
        ok = True
    if ok and shrink:
        sized = {n: max(fifo_depth, occ) for n, occ in sat.fifo_max.items() if n != "skip FIFO"}
        small = simulate_dn(r, inject_interval=0, skip_depth=r.skip_fifo.depth_words, fifo_depth=depth, elastic_map=emap,
                            fifo_depths=sized, frames=3)
        if not small.deadlock and small.steady_cyc_px <= T * (1 + tol):
            sat = small
    graph = sat.graph
    for name, occ in sat.fifo_max.items():
        graph["fifos"][name]["max_occ"] = occ
    r.fifo_graph = graph
    r.verification = dict(
        ok=ok, steady_cyc_px=sat.steady_cyc_px, latency_first_out=paced.latency_first_out, frame_cycles=sat.cycles,
        skip_needed_words=need, elastic_depth=elastic, uniform_depth=depth, tries=tries, deadlock=sat.deadlock,
        frame_periods=sat.frame_periods,
    )
    finalize_fifo_costs(r, fifo_mem)
    if not ok:
        raise RuntimeError(f"simulation does not reach T_out={T:.1f}: steady {sat.steady_cyc_px:.2f} cyc/px, deadlock={sat.deadlock}")
    return r.verification


# ---------------------------------------------------------------- rate mismatch report

def _branches(r: BottleneckResult) -> dict:
    p = r.params
    names = {x.name for x in r.nodes}
    main = ["SWG_r", "MVAU_r", "Thr_r", "FMPad", "SWG_m", "MVAU_m", "Thr_m", "MVAU_e", "Thr_e"]
    pool = ["MaxPool"] if p.get("pool_impl", "streaming") == "streaming" else ["SWG_p", "Pool"]
    if p["skip_pad"] == "mvau":
        skip = pool + ["MVAU_s", "Thr_s"]
    elif p["skip_order"] == "thr_pad":
        skip = pool + ["Thr_s", "FMPad_c"]
    else:
        skip = pool + ["FMPad_c", "Thr_s"]
    return dict(shared=["Dup"], main=[n for n in main if n in names], skip=[n for n in skip if n in names], join=["Add", "Thr_out"])


def rate_report(r: BottleneckResult, with_sim: bool = True) -> str:
    """Per node: cycles per output pixel, utilisation of the budget (model) and, from a saturated simulation, the fraction of
    cycles busy / starved / blocked. Then the mismatch between consecutive main nodes and between the two branches."""
    from dn_bottleneck_sim import simulate_dn

    T = r.params["T"]
    n = {x.name: x for x in r.nodes}
    fr = {}
    if with_sim:
        if not r.verification:
            verify_with_sim(r)
        depths = {k: f["depth"] for k, f in r.fifo_graph["fifos"].items()}
        sim = simulate_dn(r, inject_interval=0, skip_depth=r.skip_fifo.depth_words, fifo_depth=2, fifo_depths=depths)
        fr = {name: sim.fractions(name, sim.steady_window) for name in sim.node_names if name in n}
    br = _branches(r)
    lines = [f"rate mismatch, budget T_out = {T:.1f} cyc/output pixel (F = {r.params['F']} cycles/frame)"]
    hdr = f"{'node':9s} {'cyc/px':>7s} {'util':>5s} {'vs prev':>8s}" + ("   busy starv block" if fr else "")
    for title, key in (("shared input", "shared"), ("MAIN branch", "main"), ("SKIP branch", "skip"), ("JOIN", "join")):
        lines += ["", f"-- {title}", hdr]
        prev = None
        for name in br[key]:
            x = n[name]
            ratio = f"{x.cyc_px / prev:7.2f}x" if prev else "       -"
            row = f"{name:9s} {x.cyc_px:7.2f} {x.cyc_px / T:5.2f} {ratio}"
            if fr:
                f = fr[name]
                row += f"   {f['busy']:4.2f}  {f['starved']:4.2f}  {f['blocked']:4.2f}"
            lines.append(row)
            prev = x.cyc_px
    mrate = max(n[x].cyc_px for x in br["main"])
    srate = max(n[x].cyc_px for x in br["skip"])
    m_slow = max(br["main"], key=lambda x: n[x].cyc_px)
    s_slow = max(br["skip"], key=lambda x: n[x].cyc_px)
    mv = [x for x in br["main"] if x.startswith("MVAU")]
    lines += [
        "", "-- mismatch summary",
        f"main branch: slowest node {m_slow} at {mrate:.2f} cyc/px; MVAU spread {min(n[x].cyc_px for x in mv):.0f}..{max(n[x].cyc_px for x in mv):.0f} "
        f"(fastest MVAU is {min(n[x].cyc_px for x in mv) / mrate:.0%} of the slowest, idles the rest)",
        f"skip branch: slowest node {s_slow} at {srate:.2f} cyc/px (excluding the shared Dup)",
        f"main vs skip: the skip branch could deliver a pixel every {srate:.1f} cycles, the main branch every {mrate:.1f}: "
        f"skip is {mrate / srate:.1f}x faster than needed, so it is held back by backpressure and its FIFO fills "
        f"(skip FIFO {r.skip_fifo.depth_words} words x {r.skip_fifo.width_bits} bit)",
        f"shared Dup: {n['Dup'].cyc_px:.2f} cyc/px ({n['Dup'].cyc_px / T:.0%} of T) feeds both branches; input side T_in = {r.params['T_in']:.2f} cyc/input pixel",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- ONNX export (same style as bottleneck.export_onnx)

_OP_LABEL = {
    "Dup": "DuplicateStreams_hls", "SWG_r": "ConvolutionInputGenerator_rtl", "SWG_m": "ConvolutionInputGenerator_rtl",
    "MVAU_r": "MVAU_rtl", "MVAU_m": "MVAU_rtl", "MVAU_e": "MVAU_rtl", "MVAU_s": "MVAU_rtl", "FMPad": "FMPadding_rtl",
    "FMPad_c": "FMPadding_rtl", "MaxPool": "StreamingMaxPool_hls", "SWG_p": "ConvolutionInputGenerator_rtl", "Pool": "Pool_hls", "Add": "AddStreams_hls",
}


def export_onnx(r: BottleneckResult, path: str) -> None:
    """Verified downsampling dataflow graph for Netron: FINN op names, PE/SIMD/cycles/resources as attributes, a
    StreamingDataWidthConverter wherever the simulation inserted one, a StreamingFIFO_rtl on every edge (depth, width, max
    occupancy). FMPad_c carries the regrouped-stream attributes (ImgDim [H*W, C/s], NumChannels s, Padding right)."""
    import onnx
    from onnx import TensorProto, helper

    if not r.fifo_graph:
        raise RuntimeError("run verify_with_sim(result) before export_onnx")
    p, g = r.params, r.fifo_graph
    H, W, Ho, Wo = p["height"], p["width"], p["hout"], p["wout"]
    cin, cmid, cout, s = p["cin"], p["cmid"], p["cout"], p["pad_group"]
    by_name = {x.name: x for x in r.nodes}
    slowest = max(x.frame_cycles for x in r.nodes)
    ts_ch = cin if (p["skip_pad"] == "fmpad" and p["skip_order"] == "thr_pad") else cout
    node_shape = {
        "Dup": (cin, H, W), "SWG_r": (4 * cin, Ho, Wo), "MVAU_r": (cmid, Ho, Wo), "Thr_r": (cmid, Ho, Wo),
        "FMPad": (cmid, Ho + 2, Wo + 2), "SWG_m": (9 * cmid, Ho, Wo), "MVAU_m": (cmid, Ho, Wo), "Thr_m": (cmid, Ho, Wo),
        "MVAU_e": (cout, Ho, Wo), "Thr_e": (cout, Ho, Wo), "MaxPool": (cin, Ho, Wo), "SWG_p": (4 * cin, Ho, Wo), "Pool": (cin, Ho, Wo), "MVAU_s": (cout, Ho, Wo),
        "Thr_s": (ts_ch, Ho, Wo), "FMPad_c": (cout, Ho, Wo), "Add": (cout, Ho, Wo), "Thr_out": (cout, Ho, Wo),
    }
    ends = {"Source", "Sink"}
    fifos = {n: f for n, f in g["fifos"].items() if f["producer"] not in ends and f["consumer"] not in ends}
    shape = {"global_in": (cin, H, W)}
    onnx_nodes = []

    def out_tensor(node, fifo):
        outs = g["io"][node][1]
        return f"{node}:out" if len(outs) == 1 else f"{node}:out{outs.index(fifo)}"

    for name in g["nodes"]:
        if name in ends:
            continue
        ins, outs = g["io"][name]
        in_t = ["global_in" if g["fifos"][f]["producer"] == "Source" else out_tensor(g["fifos"][f]["producer"], f) for f in ins]
        if name in by_name:
            x = by_name[name]
            shp = node_shape[name]
            attrs = dict(
                stage="down_bottleneck", shape_CHW="x".join(map(str, shp)), pe=x.pe, simd=x.simd, cycles=int(x.frame_cycles),
                ii_cycles_per_output_pixel=float(x.cyc_px), pct_of_budget=100 * x.frame_cycles / p["F"],
                is_slowest_node=int(x.op.startswith("MVAU") and x.frame_cycles == slowest),
                lut=float(x.lut), bram18k=float(x.bram18), uram18=float(x.uram), dsp=int(x.dsp),
                in_width_bits=x.in_width_bits, out_width_bits=x.out_width_bits, act_bits=p["bits"],
            )
            if name == "FMPad_c":
                attrs.update(ImgDim=f"{Ho * Wo}x{cin // s}", NumChannels=s, SIMD=s, Padding=f"0,0,0,{(cout - cin) // s}",
                             note="channel pad: stream regrouped as [HW, C/s, s]")
            if name == "MVAU_s":
                attrs["weight_bits"] = 8
            op = _OP_LABEL.get(name, "Thresholding_rtl")
        else:
            in_bits, out_bits = g["fifos"][ins[0]]["bits"], g["fifos"][outs[0]]["bits"]
            shp = shape[in_t[0]]
            attrs = dict(stage="down_bottleneck", shape_CHW="x".join(map(str, shp)), inWidth=in_bits, outWidth=out_bits,
                         lut=float(fcm.dwc_cost(in_bits, out_bits)["total_lut"]), bram18k=0.0, uram18=0.0, dsp=0)
            op = "StreamingDataWidthConverter_rtl"
        wired = []
        for fname, tin in zip(ins, in_t):
            if fname not in fifos:
                wired.append(tin)
                continue
            f = fifos[fname]
            depth, bits = f["depth"], f["bits"]
            onnx_nodes.append(helper.make_node(
                "StreamingFIFO_rtl", [tin], [f"{fname}:out"], name=fname, domain="finn_milp", stage="down_bottleneck",
                shape_CHW="x".join(map(str, shape[tin])), depth=int(depth), width_bits=int(bits), bits=int(depth * bits),
                max_occupancy=int(f.get("max_occ", 0)), bram18_if_block=int(_fifo_bram18(bits, depth)),
                lutram_luts_if_distributed=int(math.ceil(bits * depth / 64)), is_skip_fifo=int(fname == "skip FIFO"),
                producer=f["producer"], consumer=f["consumer"], **_fifo_attrs(r, fname),
            ))
            shape[f"{fname}:out"] = shape[tin]
            wired.append(f"{fname}:out")
        out_names = [out_tensor(name, o) for o in outs]
        onnx_nodes.append(helper.make_node(op, wired, out_names, name=name, domain="finn_milp", **attrs))
        for t in out_names:
            shape[t] = shp
    final_out = out_tensor("Thr_out", g["io"]["Thr_out"][1][0])
    info = lambda t: helper.make_tensor_value_info(t, TensorProto.FLOAT, [1, *shape[t]])
    value_infos = [info(t) for t in shape if t not in ("global_in", final_out)]
    v = r.verification
    summary = (
        f"down bottleneck Cin={cin} Cmid={cmid} Cout={cout} {H}x{W}->{Ho}x{Wo} INT{p['bits']} T_out={p['T']:.1f} F={p['F']} "
        f"skip_pad={p['skip_pad']}({p['skip_order']}, s={s}) | LUT {r.totals['lut']:.0f} BRAM18 {r.totals['bram18']:.1f} DSP {r.totals['dsp']:.0f} | "
        f"sim steady {v.get('steady_cyc_px', float('nan')):.2f} cyc/px, latency {v.get('latency_first_out')} cycles"
    )
    graph = helper.make_graph(onnx_nodes, "down_bottleneck_dataflow", [info("global_in")], [info(final_out)],
                              value_info=value_infos, doc_string=summary)
    model = helper.make_model(graph, producer_name="dn_bottleneck_analytical",
                              opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("finn_milp", 1)])
    onnx.save(model, path)



# ---------------------------------------------------------------- FINN folding / FIFO config for hardware probes

_ROLE_OF = {
    "Dup": "dup", "SWG_r": "swg_r", "MVAU_r": "mvau_r", "Thr_r": "thr_r", "FMPad": "fmpad", "SWG_m": "swg_m", "MVAU_m": "mvau_m",
    "Thr_m": "thr_m", "MVAU_e": "mvau_e", "Thr_e": "thr_e", "MaxPool": "maxpool", "SWG_p": "swg_p", "Pool": "pool", "MVAU_s": "mvau_s", "Thr_s": "thr_s",
    "FMPad_c": "fmpad_c", "Add": "add", "Thr_out": "thr_out",
}


def to_folding_config(r: BottleneckResult) -> dict:
    """Role-keyed FINN nodeattrs, the verified FIFO depths per edge and the model's prediction (same schema as
    bottleneck.to_folding_config; roles: dup, swg_r, mvau_r, thr_r, fmpad, swg_m, mvau_m, thr_m, mvau_e, thr_e, maxpool,
    [mvau_s], thr_s, [fmpad_c], add, thr_out). fmpad_c is the channel pad: its attributes describe the REGROUPED stream
    (ImgDim [H*W, C/s], NumChannels = SIMD = s, Padding right) and are applied by hardware/finn_channel_pad.py ("custom": true).
    Needs verify_with_sim(r) to have run."""
    if not r.fifo_graph:
        raise RuntimeError("run verify_with_sim(result) before to_folding_config")
    p = r.params
    n = {x.name: x for x in r.nodes}
    cin, cmid, cout, s = p["cin"], p["cmid"], p["cout"], p["pad_group"]
    block = 1024  # depth_trigger_bram for ram_style "block"
    folding = {
        "dup": {"PE": n["Dup"].pe},
        "swg_r": {"SIMD": n["SWG_r"].simd, "parallel_window": int(n["MVAU_r"].simd > cin)},
        "mvau_r": {"PE": n["MVAU_r"].pe, "SIMD": n["MVAU_r"].simd},
        "thr_r": {"PE": n["Thr_r"].pe, "depth_trigger_bram": block},
        "fmpad": {"SIMD": n["SWG_m"].simd},
        "swg_m": {"SIMD": n["SWG_m"].simd, "parallel_window": int(n["MVAU_m"].simd > cmid)},
        "mvau_m": {"PE": n["MVAU_m"].pe, "SIMD": n["MVAU_m"].simd},
        "thr_m": {"PE": n["Thr_m"].pe, "depth_trigger_bram": block},
        "mvau_e": {"PE": n["MVAU_e"].pe, "SIMD": n["MVAU_e"].simd},
        "thr_e": {"PE": n["Thr_e"].pe, "depth_trigger_bram": block},
        **({"maxpool": {}} if p.get("pool_impl", "streaming") == "streaming" else
           {"swg_p": {"SIMD": n["SWG_p"].simd, "parallel_window": 0}, "pool": {"PE": n["Pool"].pe}}),
        "thr_s": {"PE": n["Thr_s"].pe, "depth_trigger_bram": block},
        "add": {"PE": n["Add"].pe},
        "thr_out": {"PE": n["Thr_out"].pe, "depth_trigger_bram": block},
    }
    if "MVAU_s" in n:
        folding["mvau_s"] = {"PE": n["MVAU_s"].pe, "SIMD": n["MVAU_s"].simd}
    if "FMPad_c" in n:
        folding["fmpad_c"] = {
            "custom": True, "SIMD": s, "NumChannels": s, "ImgDim": [p["hout"] * p["wout"], cin // s],
            "Padding": [0, 0, 0, (cout - cin) // s],
        }
    role = lambda name: "dwc" if name.startswith("DWC(") else _ROLE_OF.get(name, name.lower())
    fifos = []
    for name, f in r.fifo_graph["fifos"].items():
        if f["producer"] == "Source" or f["consumer"] == "Sink":
            continue
        fifos.append(dict(
            name=name, producer=role(f["producer"]), consumer=role(f["consumer"]), producer_node=f["producer"],
            consumer_node=f["consumer"], depth=int(f["depth"]), width_bits=int(f["bits"]), max_occupancy=int(f.get("max_occ", 0)),
            is_skip=name == "skip FIFO", **_fifo_attrs(r, name),
        ))
    v = r.verification
    return dict(
        params=dict(p), folding=folding, fifos=fifos,
        predicted=dict(
            T=p["T"], F=p["F"], steady_cyc_px=v.get("steady_cyc_px"), latency_first_out=v.get("latency_first_out"),
            frame_cycles=v.get("frame_cycles"), skip_fifo_words=r.skip_fifo.depth_words, skip_fifo_width_bits=r.skip_fifo.width_bits,
            totals=dict(r.totals),
            nodes={_ROLE_OF[x.name]: dict(
                op=x.op, pe=x.pe, simd=x.simd, cyc_px=x.cyc_px, frame_cycles=x.frame_cycles, lut=x.lut, bram18=x.bram18,
                uram=x.uram, dsp=x.dsp, in_width_bits=x.in_width_bits, out_width_bits=x.out_width_bits) for x in r.nodes},
            dwcs=[dict(edge=d.edge, in_width=d.in_width, out_width=d.out_width, lut=d.lut) for d in r.dwcs],
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("cin", "cout", "v", "bits", "height", "width"):
        ap.add_argument(f"--{a}", type=int, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--F", type=int, help="frame cycles budget")
    g.add_argument("--T-out", type=float, help="cycles per output pixel")
    ap.add_argument("--pad-group", type=int, help="s: channels per FMPad word (default: largest common divisor)")
    ap.add_argument("--skip-order", choices=SKIP_ORDERS, default="thr_pad")
    ap.add_argument("--skip-pad", choices=SKIP_PADS, default="fmpad")
    ap.add_argument("--pool-impl", choices=("streaming", "swg_pool"), default="streaming")
    ap.add_argument("--verify", action="store_true", help="run the cycle-level simulation check")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--folding-json", metavar="PATH", help="verify, then write the FINN probe folding + FIFO config")
    ap.add_argument("--rates", action="store_true", help="print the rate-mismatch report (model + saturated simulation)")
    ap.add_argument("--onnx", metavar="PATH", help="verify with the simulator, then write the dataflow graph (needs onnx)")
    a = ap.parse_args()
    r = model_dn_bottleneck(a.cin, a.cout, a.v, a.bits, a.height, a.width, a.F, a.T_out, a.pad_group, a.skip_order, a.skip_pad, a.pool_impl)
    if a.verify or a.rates or a.onnx or a.folding_json:
        verify_with_sim(r)
    if a.folding_json:
        with open(a.folding_json, "w") as fh:
            json.dump(to_folding_config(r), fh, indent=2)
    if a.onnx:
        export_onnx(r, a.onnx)
    print(json.dumps(r.to_dict(), indent=2) if a.json else r.report())
    if a.rates:
        print()
        print(rate_report(r))
    if a.onnx:
        print(f"wrote {a.onnx}")


if __name__ == "__main__":
    main()
