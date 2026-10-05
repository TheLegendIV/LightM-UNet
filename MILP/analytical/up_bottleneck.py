"""Analytical model of one ENet UPSAMPLING bottleneck (nearest-neighbour decoder) on FINN (companion of dn_bottleneck.py).

    main:  Dup -> MVAU_p (1x1) -> Thr_p -> UpNN (x2 nearest) -> [FMPad_k -> SWG_k -> MVAU_k (3x3) -> Thr_k] -> FIFO main -+
    ext:   Dup -> MVAU_r (1x1) -> Thr_r -> FMPadPix -> SWG_u (2x2) -> MVAU_u -> Thr_u -> MVAU_e (1x1) -> Thr_e -> skip FIFO -+-> Add -> Thr_out

skip_conv=True is decoder_type "nearest_conv_upsample" (the [..] 3x3 skip_resize_conv present), skip_conv=False "nearest_upsample"
(UpNN feeds Thr_s directly). Input H x W, output 2H x 2W; Cmid = Cin / v (FINNUpsamplingBottleneck: internal_channels = in_channels // ratio).

Differences to the regular and downsampling blocks:
* Both branches are real compute (the "main" branch holds a projection and, with skip_conv, a 3x3 conv at OUTPUT resolution: 9*Cout^2 MACs
  per output pixel, usually the heaviest MVAU of the block). The deep FIFO sits on whichever branch is shorter (usually the ext branch,
  named "skip FIFO"); "FIFO main" is the other join FIFO. Both are sized by the simulation.
* Two pixel domains again: Dup, MVAU_p, MVAU_r (and their thresholds) run on the H x W input, UpNN and everything after on the 2H x 2W output.
  Balance is on FRAME cycles: pixels_node * cyc_per_pixel <= F, reported cyc_px is per OUTPUT pixel.
* The 2x2 stride-2 transposed conv is lowered to zero-insertion (FMPadding_Pixel) + a 2x2 stride-1 conv on the (2H+1) x (2W+1) image:
  MVAU_u does 4*Cmid*Cmid MACs per OUTPUT pixel although 3 of 4 inputs of each window are inserted zeros.
* UpsampleNearestNeighbour is not foldable (all channels per word, one OUTPUT pixel per cycle: Hout*Wout cycles per frame, a floor on F).

Run: python3 up_bottleneck.py --cin 32 --cout 16 --v 4 --bits 4 --height 32 --width 32 --T-out 18 [--verify] [--rates] [--onnx PATH]
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
    BottleneckResult, NodeResult, SkipFifo, _SIZING_SCHEDULE, _dwc, _fifo_attrs, _fifo_bram18, _geom, _min_pe, _retarget_threshold,
    _search_mvau, finalize_fifo_costs,
)

fcm = reg.fcm


def _pe_at_least(channels: int, pe: int, floor: int) -> int:
    """Smallest divisor of `channels` that is >= max(pe, floor)."""
    return next(d for d in fcm.divisors(channels) if d >= max(pe, floor))


def _join_pe_for_upnn(cout: int, t_in: float, c_p_in: float, res) -> int:
    """Smallest PE | Cout with 2 * (Cout / PE) <= t_in - c_p_in (see model_up_bottleneck). PE = Cout plus a warning when no PE is enough."""
    slack = t_in - c_p_in
    for pe in fcm.divisors(cout):
        if 2 * (cout // pe) <= slack + 1e-9:
            return pe
    res.warnings.append(f"UpsampleNearestNeighbour: t_in - c_p = {slack:.1f} cyc per input pixel is below 2 even at PE = Cout; the join chain cannot drain the "
                        "re-emit phase in time. Add run-ahead buffering (>= half an input row in front of the upsampler or one output row behind it).")
    return cout


def model_up_bottleneck(
    cin: int, cout: int, v: int, bits: int, height: int, width: int, F: int | None = None, T_out: float | None = None,
    skip_conv: bool = True, join_pe: int | None = None,
) -> BottleneckResult:
    """cin -> cout upsampling block, Cmid = cin/v, INPUT map height x width, output 2*height x 2*width, uniform INT `bits`.
    Budget: F frame cycles, or T_out cycles per OUTPUT pixel (F = T_out * 4*height*width).
    join_pe: lower bound on the PE of the output-resolution join chain (Thr_s of the noconv decoder, Add, Thr_out). None (default) derives it from the budget
    for the noconv decoder (rule below) and uses the narrowest PE that fits F for the conv decoder; an int overrides.

    Why the noconv join chain needs a PE above the 'narrowest that fits F': UpsampleNearestNeighbour reads an input row and emits every pixel twice (phase 1,
    input-limited), then re-emits the stored row with no new input (phase 2, output-limited). Per input row, with no run-ahead buffering in front of it,
        W * c_p  +  2 * W * d   <=   W * t_in      <=>      d <= (t_in - c_p) / 2
    where t_in = F / (H*W) is the budget per INPUT pixel, c_p the cycles per input pixel of the slowest main-branch node in front of it (MVAU_p), and
    d = Cout / PE the cycles per OUTPUT pixel of the node behind the upsampler (Thr_s). Mean-rate balance only asks d <= t_in / 4. U4 up4: t_in 72, c_p 64 ->
    d <= 4 -> PE >= 4 (measured: PE 1 20.53, PE 2 17.96, PE 4 16.25 cyc/px, target 18). Tightening the target shrinks t_in - c_p and raises the PE; when
    t_in - c_p < 2 even PE = Cout is not enough and only buffering (>= half an input row in front of the upsampler or one output row behind) can help."""
    if cin % v:
        raise ValueError(f"Cin={cin} not divisible by v={v}")
    if (F is None) == (T_out is None):
        raise ValueError("give exactly one of F (frame cycles) or T_out (cycles per output pixel)")
    H, W, Ho, Wo = height, width, 2 * height, 2 * width
    px_in, px_out = H * W, Ho * Wo
    F = int(F if F is not None else math.floor(T_out * px_out))
    T, T_in = F / px_out, F / px_in
    cmid, A = cin // v, bits
    res = BottleneckResult(params=dict(
        cin=cin, cmid=cmid, cout=cout, v=v, z=v, T=T, F=F, T_in=T_in, bits=bits, k=3, dilation=1, stride=1, height=H, width=W,
        hout=Ho, wout=Wo, pad=1, block="up", skip_conv=skip_conv, skip_pad="n/a", skip_order="n/a", pad_group=None,
    ))
    if px_out > F:
        raise ValueError(f"F={F} is below the UpsampleNearestNeighbour floor of {px_out} cycles/frame (one output pixel per cycle, not foldable)")

    g_p = _geom("proj", cin, cout, H, W, H, W)
    g_r = _geom("reduce", cin, cmid, H, W, H, W)
    g_u = _geom("up", cmid, cmid, Ho + 1, Wo + 1, Ho, Wo, k=2, s=1, d=1, p=0)       # lowered ConvTranspose (K=S=2): conv on the (2H+1)^2 image
    g_e = _geom("expand", cmid, cout, Ho, Wo, Ho, Wo)
    g_k = _geom("skipconv", cout, cout, Ho, Wo, Ho, Wo, k=3, s=1, d=1, p=1)
    g_dup = _geom("dup", cin, cin, H, W, H, W, op="Dup")
    g_upnn = _geom("upnn", cout, cout, H, W, Ho, Wo, op="Upsample")
    g_join = _geom("join", cout, cout, Ho, Wo, Ho, Wo, op="Thresholding")

    pe_p, simd_p, c_p = _search_mvau(g_p, bits, F)
    if join_pe is None:
        join_pe = 1 if skip_conv else _join_pe_for_upnn(cout, T_in, c_p["mvu_cycles"] / px_in, res)
    res.params["join_pe"] = join_pe
    pe_r, simd_r, c_r = _search_mvau(g_r, bits, F)
    # FMPadding_Pixel over the (2H+1) x (2W+1) zero-inserted image emits cf words per pixel: it must also fit F
    pe_u, simd_u, c_u = _search_mvau(g_u, bits, F, extra_ok=lambda c: (Ho + 1) * (Wo + 1) * math.ceil(cmid / c["simd_swu"]) <= F)
    pe_e, simd_e, c_e = _search_mvau(g_e, bits, F)
    tpe_p, tpe_r = _min_pe(cout, px_in, F, "Thr_p"), _min_pe(cmid, px_in, F, "Thr_r")
    tpe_u, tpe_e = _min_pe(cmid, px_out, F, "Thr_u"), _min_pe(cout, px_out, F, "Thr_e")
    c_p, c_r = _retarget_threshold(c_p, g_p, A, tpe_p), _retarget_threshold(c_r, g_r, A, tpe_r)
    c_u, c_e = _retarget_threshold(c_u, g_u, A, tpe_u), _retarget_threshold(c_e, g_e, A, tpe_e)
    if skip_conv:
        pe_k, simd_k, c_k = _search_mvau(g_k, bits, F)
        tpe_k = _min_pe(cout, px_out, F, "Thr_k")
        c_k = _retarget_threshold(c_k, g_k, A, tpe_k)
    else:
        pe_ts = _pe_at_least(cout, _min_pe(cout, px_out, F, "Thr_s"), join_pe)
        c_ts = fcm.threshold_node_cost(g_join, A, pe_ts)

    pe_d = _min_pe(cin, px_in, F, "Dup")
    pe_a = _pe_at_least(cout, _min_pe(cout, px_out, F, "Add"), join_pe)
    pe_to = _pe_at_least(cout, _min_pe(cout, px_out, F, "Thr_out"), join_pe)
    c_dup = fcm.stream_node_cost("dup", g_dup, pe_d)
    c_up = fcm.stream_node_cost("upsample", g_upnn)
    c_add = fcm.stream_node_cost("add", g_join, pe_a)
    c_to = fcm.threshold_node_cost(g_join, A, pe_to)
    add_bits = A + 1
    acc = {k: c["acc_bits"] for k, c in (("p", c_p), ("r", c_r), ("u", c_u), ("e", c_e))}
    swu_u, cf_u = c_u["simd_swu"], math.ceil(cmid / c_u["simd_swu"])
    par_u = simd_u > cmid                                 # parallel_window: one 2x2 window per SWG word, a DWC narrows it to the MVAU's SIMD
    pad_pix_cycles = (Ho - 1) * (Wo - 1) * cf_u          # FMPadding_Pixel: zero insertion only, (2H-1)^2 grid
    pad_edge_cycles = (Ho + 1) * (Wo + 1) * cf_u         # FMPadding_rtl behind it: the pad-1 border, (2H+1)^2 grid

    def row(name, op, pe, simd, frame_cycles, lut=0.0, bram=0.0, uram=0.0, dsp=0.0, in_w=0, out_w=0):
        return NodeResult(name=name, op=op, pe=pe, simd=simd, cyc_px=frame_cycles / px_out, frame_cycles=int(frame_cycles),
                          lut=lut, bram18=bram, uram=uram, dsp=dsp, in_width_bits=in_w, out_width_bits=out_w)

    nd = res.nodes
    nd.append(row("Dup", "DuplicateStreams", pe_d, 0, c_dup["cycles"], c_dup["total_lut"], in_w=pe_d * A, out_w=pe_d * A))
    nd.append(row("MVAU_p", "MVAU rtl 1x1 proj", pe_p, simd_p, c_p["mvu_cycles"], c_p["mvu_lut"], c_p["wm_bram18"], c_p["wm_uram18"],
                  c_p["mvu_dsp"], in_w=simd_p * A, out_w=pe_p * acc["p"]))
    nd.append(row("Thr_p", "Thresholding_rtl", tpe_p, 0, px_in * (cout // tpe_p), c_p["thr_lut"], c_p["thr_bram18"], in_w=tpe_p * acc["p"], out_w=tpe_p * A))
    nd.append(row("UpNN", "UpsampleNearestNeighbour_hls", 0, 0, c_up["cycles"], c_up["total_lut"], in_w=cout * A, out_w=cout * A))
    if skip_conv:
        simd_swu_k = c_k["simd_swu"]
        nd.append(row("FMPad_k", "FMPadding 3x3", 0, simd_swu_k, c_k["fmpad_cycles"], in_w=simd_swu_k * A, out_w=simd_swu_k * A))
        nd.append(row("SWG_k", "ConvolutionInputGenerator_rtl", 0, simd_swu_k, c_k["swu_cycles"], c_k["swu_lut"], c_k["swu_bram18"], c_k["swu_uram18"],
                      in_w=simd_swu_k * A, out_w=simd_swu_k * A))
        nd.append(row("MVAU_k", "MVAU rtl 3x3 skip conv", pe_k, simd_k, c_k["mvu_cycles"], c_k["mvu_lut"], c_k["wm_bram18"], c_k["wm_uram18"],
                      c_k["mvu_dsp"], in_w=simd_k * A, out_w=pe_k * c_k["acc_bits"]))
        nd.append(row("Thr_k", "Thresholding_rtl", tpe_k, 0, px_out * (cout // tpe_k), c_k["thr_lut"], c_k["thr_bram18"],
                      in_w=tpe_k * c_k["acc_bits"], out_w=tpe_k * A))
        main_out_pe = tpe_k
    else:
        nd.append(row("Thr_s", "Thresholding_rtl (main)", pe_ts, 0, c_ts["cycles"], c_ts["total_lut"], c_ts["thr_bram18"], in_w=pe_ts * A, out_w=pe_ts * A))
        main_out_pe = pe_ts
    nd.append(row("MVAU_r", "MVAU rtl 1x1 reduce", pe_r, simd_r, c_r["mvu_cycles"], c_r["mvu_lut"], c_r["wm_bram18"], c_r["wm_uram18"],
                  c_r["mvu_dsp"], in_w=simd_r * A, out_w=pe_r * acc["r"]))
    nd.append(row("Thr_r", "Thresholding_rtl", tpe_r, 0, px_in * (cmid // tpe_r), c_r["thr_lut"], c_r["thr_bram18"], in_w=tpe_r * acc["r"], out_w=tpe_r * A))
    nd.append(row("FMPadPix", "FMPadding_Pixel_hls", 0, swu_u, pad_pix_cycles, in_w=swu_u * A, out_w=swu_u * A))
    nd.append(row("FMPad_u", "FMPadding_rtl (pad 1)", 0, swu_u, pad_edge_cycles, in_w=swu_u * A, out_w=swu_u * A))
    nd.append(row("SWG_u", "ConvolutionInputGenerator_rtl 2x2" + (" parallel_window" if par_u else ""), 0, swu_u, c_u["swu_cycles"], c_u["swu_lut"],
                  c_u["swu_bram18"], c_u["swu_uram18"], in_w=swu_u * A, out_w=swu_u * A * (4 if par_u else 1)))
    nd.append(row("MVAU_u", "MVAU rtl 2x2 (lowered ConvTranspose)", pe_u, simd_u, c_u["mvu_cycles"], c_u["mvu_lut"], c_u["wm_bram18"], c_u["wm_uram18"],
                  c_u["mvu_dsp"], in_w=simd_u * A, out_w=pe_u * acc["u"]))
    nd.append(row("Thr_u", "Thresholding_rtl", tpe_u, 0, px_out * (cmid // tpe_u), c_u["thr_lut"], c_u["thr_bram18"], in_w=tpe_u * acc["u"], out_w=tpe_u * A))
    nd.append(row("MVAU_e", "MVAU rtl 1x1 expand", pe_e, simd_e, c_e["mvu_cycles"], c_e["mvu_lut"], c_e["wm_bram18"], c_e["wm_uram18"],
                  c_e["mvu_dsp"], in_w=simd_e * A, out_w=pe_e * acc["e"]))
    nd.append(row("Thr_e", "Thresholding_rtl", tpe_e, 0, px_out * (cout // tpe_e), c_e["thr_lut"], c_e["thr_bram18"], in_w=tpe_e * acc["e"], out_w=tpe_e * A))
    nd.append(row("Add", "AddStreams", pe_a, 0, c_add["cycles"], c_add["total_lut"], in_w=pe_a * A, out_w=pe_a * add_bits))
    nd.append(row("Thr_out", "Thresholding_rtl (join)", pe_to, 0, c_to["cycles"], c_to["total_lut"], c_to["thr_bram18"], in_w=pe_to * add_bits, out_w=pe_to * A))

    dw = res.dwcs
    _dwc("Dup->MVAU_p", pe_d * A, simd_p * A, cin, pe_d, simd_p, px_in, px_out, dw)
    _dwc("MVAU_p->Thr_p", pe_p * acc["p"], tpe_p * acc["p"], cout, pe_p, tpe_p, px_in, px_out, dw)
    _dwc("Thr_p->UpNN", tpe_p * A, cout * A, cout, tpe_p, cout, px_in, px_out, dw)
    if skip_conv:
        _dwc("UpNN->FMPad_k", cout * A, simd_swu_k * A, cout, cout, simd_swu_k, px_out, px_out, dw)
        _dwc("MVAU_k->Thr_k", pe_k * c_k["acc_bits"], tpe_k * c_k["acc_bits"], cout, pe_k, tpe_k, px_out, px_out, dw)
    else:
        _dwc("UpNN->Thr_s", cout * A, pe_ts * A, cout, cout, pe_ts, px_out, px_out, dw)
    _dwc("FIFOmain->Add", main_out_pe * A, pe_a * A, cout, main_out_pe, pe_a, px_out, px_out, dw)
    _dwc("Dup->MVAU_r", pe_d * A, simd_r * A, cin, pe_d, simd_r, px_in, px_out, dw)
    _dwc("MVAU_r->Thr_r", pe_r * acc["r"], tpe_r * acc["r"], cmid, pe_r, tpe_r, px_in, px_out, dw)
    _dwc("Thr_r->FMPadPix", tpe_r * A, swu_u * A, cmid, tpe_r, swu_u, px_in, px_out, dw)
    if par_u:
        _dwc("SWG_u->MVAU_u", 4 * swu_u * A, simd_u * A, 4 * cmid, 4 * swu_u, simd_u, px_out, px_out, dw)
    _dwc("MVAU_u->Thr_u", pe_u * acc["u"], tpe_u * acc["u"], cmid, pe_u, tpe_u, px_out, px_out, dw)
    _dwc("Thr_u->MVAU_e", tpe_u * A, simd_e * A, cmid, tpe_u, simd_e, px_out, px_out, dw)
    _dwc("MVAU_e->Thr_e", pe_e * acc["e"], tpe_e * acc["e"], cout, pe_e, tpe_e, px_out, px_out, dw)
    _dwc("skipFIFO->Add", tpe_e * A, pe_a * A, cout, tpe_e, pe_a, px_out, px_out, dw)
    _dwc("Add->Thr_out", pe_a * add_bits, pe_to * add_bits, cout, pe_a, pe_to, px_out, px_out, dw)
    for d in dw:
        if d.cyc_px > T + 1e-9:
            res.warnings.append(f"DWC {d.edge} needs {d.cyc_px:.1f} cyc/px (output pixels) > T_out={T:.1f}")

    # ---- rough latency / join-FIFO estimate (the simulation corrects it): first pixel through each branch
    cyc = {x.name: x.cyc_px for x in nd}
    thr_px = lambda ch, pe: ch / pe
    dup_px = cin / pe_d
    l_main = dup_px + cyc["MVAU_p"] + thr_px(cout, tpe_p)
    if skip_conv:
        cf_k = math.ceil(cout / simd_swu_k)
        l_main += (W - 1) * T_in + (Wo + 2) * cf_k + 9 * cout / simd_k + cyc["MVAU_k"] + thr_px(cout, tpe_k)   # 3x3 window needs ~one input row
    else:
        l_main += thr_px(cout, pe_ts)
    l_ext = dup_px + cyc["MVAU_r"] + thr_px(cmid, tpe_r) + (Wo + 2) * cf_u + 4 * cmid / simd_u + cyc["MVAU_u"] + thr_px(cmid, tpe_u) + cyc["MVAU_e"] + thr_px(cout, tpe_e)
    ext_w, main_w = cout // tpe_e, cout // main_out_pe
    skip_px = max(1, math.ceil(max(0.0, l_main - l_ext) / T)) + 1
    depth = skip_px * ext_w
    fw = tpe_e * A
    res.skip_fifo = SkipFifo(width_bits=fw, depth_words=depth, bits=fw * depth, bram18_if_block=_fifo_bram18(fw, depth),
                             lutram_luts_if_distributed=math.ceil(fw * depth / 64), pixels_buffered=skip_px, pe=tpe_e)
    res.params["main_fifo_words"] = max(2, (max(1, math.ceil(max(0.0, l_ext - l_main) / T)) + 1) * main_w)
    res.latency_first_out_cycles = int(math.ceil(max(l_main, l_ext) + cout / pe_a + cout / pe_to))
    res.frame_cycles = int(res.latency_first_out_cycles + (px_out - 1) * T)
    dwc_lut = sum(d.lut for d in dw)
    res.totals = dict(lut=sum(x.lut for x in nd) + dwc_lut, bram18=sum(x.bram18 for x in nd) + res.skip_fifo.bram18_if_block,
                      uram=sum(x.uram for x in nd), dsp=sum(x.dsp for x in nd), dwc_lut=dwc_lut)
    mv = [x for x in nd if x.op.startswith("MVAU")]
    cycs = [x.cyc_px for x in mv]
    res.balance = dict(max_cyc_px=max(cycs), min_cyc_px=min(cycs), min_over_max=min(cycs) / max(cycs), mean_util=sum(cycs) / len(cycs) / T,
                       scope="MVAU nodes")
    if max(x.frame_cycles for x in nd) > F:
        res.warnings.append(f"slowest node {max(x.frame_cycles for x in nd)} cycles/frame exceeds F={F}")
    for x in mv:
        if x.frame_cycles < F * 0.999:
            res.warnings.append(f"{x.name} runs at {x.frame_cycles / F:.0%} of F: no (PE, SIMD) divisor pair lands exactly on F")
    res.warnings.append("FMPadPix / FMPad_k / UpNN LUT are not calibrated in finn_cost_model (priced 0 or provisional); MVAU_u counts the "
                        "MACs on inserted zeros (FINN lowers ConvTranspose to zero insertion + conv)")
    return res


def verify_with_sim(r: BottleneckResult, fifo_depth: int = 2, tol: float = 0.02, max_tries: int = 6, shrink: bool = True,
                    tol_soft: float = 0.03, fifo_mem: str = "auto") -> dict:
    """Paced 2-frame run with both join FIFOs unbounded measures what each needs; then a saturated 3-frame run escalates the FIFOs in
    front of FMPad_k / FMPadPix (and a uniform depth) until the last frame's period per OUTPUT pixel is within tol of T_out."""
    from up_bottleneck_sim import UNBOUNDED, simulate_up

    p = r.params
    T = p["T"]
    conv = p["skip_conv"]
    n = {x.name: x for x in r.nodes}
    cf_k = p["cout"] // n["SWG_k"].simd if conv else 1
    cf_u = p["cmid"] // n["SWG_u"].simd
    ext_w = p["cout"] // r.skip_fifo.pe
    main_w = p["cout"] // (n["Thr_k"].pe if conv else n["Thr_s"].pe)
    emap0 = {"FMPadPix": 2 * cf_u + 2, "FMPad_u": 2 * cf_u + 2}
    if conv:
        emap0["FMPad_k"] = (p["wout"] + 2) * cf_k + 2
    pad_out = (p["wout"] + 2) * cf_k + 2
    paced = simulate_up(r, inject_interval=p["T_in"], skip_depth=UNBOUNDED, main_depth=UNBOUNDED, fifo_depth=fifo_depth,
                        elastic_map=emap0, frames=2)
    need_skip, need_main = paced.fifo_max["skip FIFO"], paced.fifo_max["FIFO main"]
    d = need_skip + ext_w                                   # measured need + one pixel (the analytic estimate above is rough, either way)
    if d != r.skip_fifo.depth_words:
        r.warnings.append(f"skip FIFO set from the analytic {r.skip_fifo.depth_words} to {d} words by simulation")
        f = r.skip_fifo
        f.depth_words, f.bits, f.pixels_buffered = d, f.width_bits * d, math.ceil(d / ext_w)
        f.bram18_if_block, f.lutram_luts_if_distributed = _fifo_bram18(f.width_bits, d), math.ceil(f.width_bits * d / 64)
    p["main_fifo_words"] = max(2, need_main + main_w)
    sat, tries, emap = None, 0, dict(emap0)
    for tries, (es, with_pad, um) in enumerate(_SIZING_SCHEDULE[:max_tries], 1):
        depth = fifo_depth * um
        emap = {k: max(v * es, depth) for k, v in emap0.items()}
        extra = {"FMPad_k->out": pad_out * es} if (with_pad and conv) else {}
        sat = simulate_up(r, inject_interval=0, skip_depth=r.skip_fifo.depth_words, main_depth=p["main_fifo_words"], fifo_depth=depth,
                          elastic_map=emap, fifo_depths=extra, frames=3)
        if not sat.deadlock and sat.steady_cyc_px <= T * (1 + tol):
            break
    ok = (not sat.deadlock) and sat.steady_cyc_px <= T * (1 + tol)
    if ok and shrink:
        sized = {k: max(fifo_depth, occ) for k, occ in sat.fifo_max.items() if k not in ("skip FIFO", "FIFO main")}
        small = simulate_up(r, inject_interval=0, skip_depth=r.skip_fifo.depth_words, main_depth=p["main_fifo_words"], fifo_depth=depth,
                            elastic_map=emap, fifo_depths=sized, frames=3)
        if not small.deadlock and small.steady_cyc_px <= T * (1 + tol):
            sat = small
    if not ok and not sat.deadlock and sat.steady_cyc_px <= T * (1 + tol_soft):
        r.warnings.append(f"steady {sat.steady_cyc_px:.2f} cyc/px is {sat.steady_cyc_px / T - 1:.1%} above T: residual per-frame window-fill gap")
        ok = True
    graph = sat.graph
    for name, occ in sat.fifo_max.items():
        graph["fifos"][name]["max_occ"] = occ
    r.fifo_graph = graph
    r.verification = dict(
        ok=ok, steady_cyc_px=sat.steady_cyc_px, latency_first_out=paced.latency_first_out, frame_cycles=sat.cycles,
        skip_needed_words=need_skip, main_needed_words=need_main, uniform_depth=depth, tries=tries, deadlock=sat.deadlock,
        elastic_depth=emap.get("FMPad_k", emap.get("FMPadPix")),
        frame_periods=sat.frame_periods,
    )
    finalize_fifo_costs(r, fifo_mem)
    if not ok:
        raise RuntimeError(f"simulation does not reach T_out={T:.1f}: steady {sat.steady_cyc_px:.2f} cyc/px, deadlock={sat.deadlock}")
    return r.verification


# ---------------------------------------------------------------- rate mismatch report

def _branches(r: BottleneckResult) -> dict:
    main = ["MVAU_p", "Thr_p", "UpNN"] + (["FMPad_k", "SWG_k", "MVAU_k", "Thr_k"] if r.params["skip_conv"] else ["Thr_s"])
    ext = ["MVAU_r", "Thr_r", "FMPadPix", "FMPad_u", "SWG_u", "MVAU_u", "Thr_u", "MVAU_e", "Thr_e"]
    return dict(shared=["Dup"], main=main, ext=ext, join=["Add", "Thr_out"])


def rate_report(r: BottleneckResult, with_sim: bool = True) -> str:
    from up_bottleneck_sim import simulate_up

    T = r.params["T"]
    n = {x.name: x for x in r.nodes}
    fr = {}
    if with_sim:
        if not r.verification:
            verify_with_sim(r)
        depths = {k: f["depth"] for k, f in r.fifo_graph["fifos"].items()}
        sim = simulate_up(r, inject_interval=0, skip_depth=r.skip_fifo.depth_words, main_depth=r.params["main_fifo_words"], fifo_depth=2,
                          fifo_depths=depths)
        fr = {name: sim.fractions(name, sim.steady_window) for name in sim.node_names if name in n}
    br = _branches(r)
    lines = [f"rate mismatch, budget T_out = {T:.1f} cyc/output pixel (F = {r.params['F']} cycles/frame)"]
    hdr = f"{'node':9s} {'cyc/px':>7s} {'util':>5s} {'vs prev':>8s}" + ("   busy starv block" if fr else "")
    for title, key in (("shared input", "shared"), ("MAIN branch (proj -> upsample -> 3x3)", "main"), ("EXT branch (reduce -> up -> expand)", "ext"), ("JOIN", "join")):
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
    erate = max(n[x].cyc_px for x in br["ext"])
    m_slow = max(br["main"], key=lambda x: n[x].cyc_px)
    e_slow = max(br["ext"], key=lambda x: n[x].cyc_px)
    lines += ["", "-- mismatch summary",
              f"main branch slowest {m_slow} at {mrate:.2f} cyc/px; ext branch slowest {e_slow} at {erate:.2f} cyc/px; budget {T:.1f}",
              f"join FIFOs: skip FIFO (ext end) {r.skip_fifo.depth_words} words x {r.skip_fifo.width_bits} bit, FIFO main {r.params['main_fifo_words']} words",
              f"input side T_in = {r.params['T_in']:.2f} cyc/input pixel (Dup {n['Dup'].cyc_px:.2f} cyc/px on the output scale)"]
    return "\n".join(lines)


# ---------------------------------------------------------------- FINN folding / FIFO config

_ROLE_OF = {
    "Dup": "dup", "MVAU_p": "mvau_p", "Thr_p": "thr_p", "UpNN": "upnn", "FMPad_k": "fmpad_k", "SWG_k": "swg_k", "MVAU_k": "mvau_k",
    "Thr_k": "thr_k", "Thr_s": "thr_s", "MVAU_r": "mvau_r", "Thr_r": "thr_r", "FMPadPix": "fmpadpix", "FMPad_u": "fmpad_u", "SWG_u": "swg_u", "MVAU_u": "mvau_u",
    "Thr_u": "thr_u", "MVAU_e": "mvau_e", "Thr_e": "thr_e", "Add": "add", "Thr_out": "thr_out",
}


def to_folding_config(r: BottleneckResult) -> dict:
    """Role-keyed FINN nodeattrs + verified FIFO depths per edge + the model prediction (schema of bottleneck.to_folding_config).
    Join FIFOs: 'skip FIFO' (ext branch end) and 'FIFO main'. Needs verify_with_sim(r)."""
    if not r.fifo_graph:
        raise RuntimeError("run verify_with_sim(result) before to_folding_config")
    p = r.params
    n = {x.name: x for x in r.nodes}
    cin, cmid, cout = p["cin"], p["cmid"], p["cout"]
    block = 1024
    fold = {
        "dup": {"PE": n["Dup"].pe},
        "mvau_p": {"PE": n["MVAU_p"].pe, "SIMD": n["MVAU_p"].simd}, "thr_p": {"PE": n["Thr_p"].pe, "depth_trigger_bram": block},
        "upnn": {},
        "mvau_r": {"PE": n["MVAU_r"].pe, "SIMD": n["MVAU_r"].simd}, "thr_r": {"PE": n["Thr_r"].pe, "depth_trigger_bram": block},
        "fmpadpix": {"SIMD": n["SWG_u"].simd},
        "fmpad_u": {"SIMD": n["SWG_u"].simd},
        "swg_u": {"SIMD": n["SWG_u"].simd, "parallel_window": int(n["MVAU_u"].simd > cmid)},
        "mvau_u": {"PE": n["MVAU_u"].pe, "SIMD": n["MVAU_u"].simd}, "thr_u": {"PE": n["Thr_u"].pe, "depth_trigger_bram": block},
        "mvau_e": {"PE": n["MVAU_e"].pe, "SIMD": n["MVAU_e"].simd}, "thr_e": {"PE": n["Thr_e"].pe, "depth_trigger_bram": block},
        "add": {"PE": n["Add"].pe}, "thr_out": {"PE": n["Thr_out"].pe, "depth_trigger_bram": block},
    }
    if p["skip_conv"]:
        fold.update({
            "fmpad_k": {"SIMD": n["SWG_k"].simd},
            "swg_k": {"SIMD": n["SWG_k"].simd, "parallel_window": int(n["MVAU_k"].simd > cout)},
            "mvau_k": {"PE": n["MVAU_k"].pe, "SIMD": n["MVAU_k"].simd}, "thr_k": {"PE": n["Thr_k"].pe, "depth_trigger_bram": block},
        })
    else:
        fold["thr_s"] = {"PE": n["Thr_s"].pe, "depth_trigger_bram": block}
    role = lambda name: "dwc" if name.startswith("DWC(") else _ROLE_OF.get(name, name.lower())
    fifos = []
    for name, f in r.fifo_graph["fifos"].items():
        if f["producer"] == "Source" or f["consumer"] == "Sink":
            continue
        fifos.append(dict(name=name, producer=role(f["producer"]), consumer=role(f["consumer"]), producer_node=f["producer"],
                          consumer_node=f["consumer"], depth=int(f["depth"]), width_bits=int(f["bits"]), max_occupancy=int(f.get("max_occ", 0)),
                          is_skip=name == "skip FIFO", is_join_fifo=name in ("skip FIFO", "FIFO main"), **_fifo_attrs(r, name)))
    v = r.verification
    return dict(
        params=dict(p), folding=fold, fifos=fifos,
        predicted=dict(
            T=p["T"], F=p["F"], steady_cyc_px=v.get("steady_cyc_px"), latency_first_out=v.get("latency_first_out"), frame_cycles=v.get("frame_cycles"),
            skip_fifo_words=r.skip_fifo.depth_words, skip_fifo_width_bits=r.skip_fifo.width_bits, main_fifo_words=p["main_fifo_words"],
            totals=dict(r.totals),
            nodes={_ROLE_OF[x.name]: dict(op=x.op, pe=x.pe, simd=x.simd, cyc_px=x.cyc_px, frame_cycles=x.frame_cycles, lut=x.lut, bram18=x.bram18,
                                          uram=x.uram, dsp=x.dsp, in_width_bits=x.in_width_bits, out_width_bits=x.out_width_bits) for x in r.nodes},
            dwcs=[dict(edge=d.edge, in_width=d.in_width, out_width=d.out_width, lut=d.lut) for d in r.dwcs],
        ),
    )


# ---------------------------------------------------------------- ONNX export

_OP_LABEL = {
    "Dup": "DuplicateStreams_hls", "MVAU_p": "MVAU_rtl", "MVAU_r": "MVAU_rtl", "MVAU_u": "MVAU_rtl", "MVAU_e": "MVAU_rtl", "MVAU_k": "MVAU_rtl",
    "UpNN": "UpsampleNearestNeighbour_hls", "FMPad_k": "FMPadding_rtl", "FMPadPix": "FMPadding_Pixel_hls", "FMPad_u": "FMPadding_rtl",
    "SWG_k": "ConvolutionInputGenerator_rtl", "SWG_u": "ConvolutionInputGenerator_rtl", "Add": "AddStreams_hls",
}


def export_onnx(r: BottleneckResult, path: str) -> None:
    """Verified upsampling dataflow graph for Netron (FINN op names, PE/SIMD/cycles/resources as attributes, DWCs, FIFOs with memory mapping)."""
    import onnx
    from onnx import TensorProto, helper

    if not r.fifo_graph:
        raise RuntimeError("run verify_with_sim(result) before export_onnx")
    p, g = r.params, r.fifo_graph
    H, W, Ho, Wo = p["height"], p["width"], p["hout"], p["wout"]
    cin, cmid, cout = p["cin"], p["cmid"], p["cout"]
    by_name = {x.name: x for x in r.nodes}
    slowest = max(x.frame_cycles for x in r.nodes)
    node_shape = {
        "Dup": (cin, H, W), "MVAU_p": (cout, H, W), "Thr_p": (cout, H, W), "UpNN": (cout, Ho, Wo), "FMPad_k": (cout, Ho + 2, Wo + 2),
        "SWG_k": (9 * cout, Ho, Wo), "MVAU_k": (cout, Ho, Wo), "Thr_k": (cout, Ho, Wo), "Thr_s": (cout, Ho, Wo), "MVAU_r": (cmid, H, W),
        "Thr_r": (cmid, H, W), "FMPadPix": (cmid, Ho - 1, Wo - 1), "FMPad_u": (cmid, Ho + 1, Wo + 1), "SWG_u": (4 * cmid, Ho, Wo), "MVAU_u": (cmid, Ho, Wo), "Thr_u": (cmid, Ho, Wo),
        "MVAU_e": (cout, Ho, Wo), "Thr_e": (cout, Ho, Wo), "Add": (cout, Ho, Wo), "Thr_out": (cout, Ho, Wo),
    }
    ends = {"Source", "Sink"}
    fifos = {k: f for k, f in g["fifos"].items() if f["producer"] not in ends and f["consumer"] not in ends}
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
            attrs = dict(stage="up_bottleneck", shape_CHW="x".join(map(str, shp)), pe=x.pe, simd=x.simd, cycles=int(x.frame_cycles),
                         ii_cycles_per_output_pixel=float(x.cyc_px), pct_of_budget=100 * x.frame_cycles / p["F"],
                         is_slowest_node=int(x.op.startswith("MVAU") and x.frame_cycles == slowest), lut=float(x.lut), bram18k=float(x.bram18),
                         uram18=float(x.uram), dsp=int(x.dsp), in_width_bits=x.in_width_bits, out_width_bits=x.out_width_bits, act_bits=p["bits"])
            op = _OP_LABEL.get(name, "Thresholding_rtl")
        else:
            in_bits, out_bits = g["fifos"][ins[0]]["bits"], g["fifos"][outs[0]]["bits"]
            shp = shape[in_t[0]]
            attrs = dict(stage="up_bottleneck", shape_CHW="x".join(map(str, shp)), inWidth=in_bits, outWidth=out_bits,
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
                "StreamingFIFO_rtl", [tin], [f"{fname}:out"], name=fname, domain="finn_milp", stage="up_bottleneck",
                shape_CHW="x".join(map(str, shape[tin])), depth=int(depth), width_bits=int(bits), bits=int(depth * bits),
                max_occupancy=int(f.get("max_occ", 0)), is_skip_fifo=int(fname == "skip FIFO"), is_join_fifo=int(fname in ("skip FIFO", "FIFO main")),
                producer=f["producer"], consumer=f["consumer"], **_fifo_attrs(r, fname)))
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
    summary = (f"up bottleneck Cin={cin} Cmid={cmid} Cout={cout} {H}x{W}->{Ho}x{Wo} INT{p['bits']} T_out={p['T']:.1f} F={p['F']} "
               f"skip_conv={p['skip_conv']} | LUT {r.totals['lut']:.0f} BRAM18 {r.totals['bram18']:.1f} URAM {r.totals['uram']:.0f} DSP {r.totals['dsp']:.0f} | "
               f"sim steady {v.get('steady_cyc_px', float('nan')):.2f} cyc/px, latency {v.get('latency_first_out')} cycles")
    graph = helper.make_graph(onnx_nodes, "up_bottleneck_dataflow", [info("global_in")], [info(final_out)], value_info=value_infos, doc_string=summary)
    model = helper.make_model(graph, producer_name="up_bottleneck_analytical",
                              opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("finn_milp", 1)])
    onnx.save(model, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("cin", "cout", "v", "bits", "height", "width"):
        ap.add_argument(f"--{a}", type=int, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--F", type=int, help="frame cycles budget")
    g.add_argument("--T-out", type=float, help="cycles per OUTPUT pixel")
    ap.add_argument("--no-skip-conv", action="store_true", help="decoder_type nearest_upsample (no 3x3 skip_resize_conv)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--rates", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--onnx", metavar="PATH")
    ap.add_argument("--folding-json", metavar="PATH")
    a = ap.parse_args()
    r = model_up_bottleneck(a.cin, a.cout, a.v, a.bits, a.height, a.width, a.F, a.T_out, skip_conv=not a.no_skip_conv)
    if a.verify or a.rates or a.onnx or a.folding_json:
        verify_with_sim(r)
    if a.onnx:
        export_onnx(r, a.onnx)
    if a.folding_json:
        with open(a.folding_json, "w") as fh:
            json.dump(to_folding_config(r), fh, indent=2)
    print(json.dumps(r.to_dict(), indent=2) if a.json else r.report())
    if a.rates:
        print()
        print(rate_report(r))
    if a.onnx:
        print(f"wrote {a.onnx}")


if __name__ == "__main__":
    main()
