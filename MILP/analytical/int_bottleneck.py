"""Analytical model of the ENet INITIAL block on FINN (companion of reg_/dn_/up_bottleneck.py), reference: FINNInitialBlockConcat in
enet/nnunetv2/nets/LayerQuantEnetFINN.py (U4 network: 1 -> 4 channels, 256x256 -> 128x128).

    Thr_in -> Dup -> [FMPad -> SWG (3x3, stride 2) -> MVAU_c (9*Cin -> Cout-Cin) -> Thr_c] -> FIFO main -+
                     [Thr_m -> MaxPool (2x2, stride 2)] ---------------------------------> skip FIFO -----+-> Concat -> Thr_act

Thr_in is the block's input quantizer, Thr_c / Thr_m the shared `branch_quant` on both branches, Thr_act the BN + ReLU after the concat.
Thr_m sits UPSTREAM of the maxpool (landed FINN graph, rtlsim probes of 2026-10-03/04): the exported ONNX has MaxPool -> Quant, but the streamline step
`MoveMaxPoolPastMultiThreshold` (hardware/finn_enet_build.py) swaps MaxPool -> MultiThreshold into MultiThreshold -> MaxPool. FINN does it to pool narrow data
behind a conv; here the pool already reads the INT-A input stream, so it only makes Thr_m process H*W pixels (4x the output count) at full resolution.
Input H x W, output (H/2) x (W/2). Balance is on FRAME cycles: pixels_node * cyc_per_pixel <= F, reported cyc_px is per OUTPUT pixel.

Facts that shape this block:
* MaxPool is not foldable (all channels per cycle, ~1.25 * H * W cycles per frame): for a 256x256 input that is 81,920 cycles, a hard floor on the
  frame budget F of the WHOLE network (the other probe blocks use F = 73,728). The model rejects a smaller F and states the floor.
* The 3x3 stride-2 conv on one input channel has MW = 9: the sliding window runs in parallel-window mode (SIMD = 9) or the window generator's 9
  words per window would exceed F.
* Concat joins the branches (not an Add): StreamingConcat, one output pixel per cycle, no folding.

Run: python3 int_bottleneck.py --cin 1 --cout 4 --bits 4 --height 256 --width 256 --F 81920 [--verify] [--rates] [--onnx PATH]
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
    _search_mvau, finalize_fifo_costs, search_swg_pool,
)

fcm = reg.fcm


def model_int_bottleneck(
    cin: int, cout: int, bits: int, height: int, width: int, F: int | None = None, T_out: float | None = None, pool_impl: str = "streaming",
) -> BottleneckResult:
    """cin -> cout initial block (cout > cin: conv branch makes cout - cin channels, maxpool branch cin), INPUT map height x width (even),
    uniform INT `bits`. Budget: F frame cycles, or T_out cycles per OUTPUT pixel (F = T_out * height/2 * width/2)."""
    if height % 2 or width % 2:
        raise ValueError("height and width must be even")
    if cout <= cin:
        raise ValueError("initial block concatenates conv (cout - cin channels) and maxpool (cin channels): need cout > cin")
    if (F is None) == (T_out is None):
        raise ValueError("give exactly one of F (frame cycles) or T_out (cycles per output pixel)")
    H, W, Ho, Wo = height, width, height // 2, width // 2
    px_in, px_out = H * W, Ho * Wo
    F = int(F if F is not None else math.floor(T_out * px_out))
    T, T_in, A, ccv = F / px_out, F / px_in, bits, cout - cin
    res = BottleneckResult(params=dict(
        cin=cin, cmid=ccv, cout=cout, v=1, z=1, T=T, F=F, T_in=T_in, bits=bits, k=3, dilation=1, stride=2, height=H, width=W, hout=Ho, wout=Wo,
        pad=1, block="init", skip_pad="n/a", skip_order="n/a", pad_group=None, pool_impl=pool_impl,
    ))

    g_mp = _geom("pool", cin, cin, H, W, Ho, Wo, k=2, s=2, op="MaxPool2d")
    if pool_impl not in ("streaming", "swg_pool"):
        raise ValueError("pool_impl in (streaming = StreamingMaxPool, swg_pool = depthwise SWG + Pool_hls with PE)")
    if pool_impl == "streaming":
        c_mp = fcm.maxpool_cost(g_mp, A)
        if c_mp["cycles"] > F:
            raise ValueError(f"F={F} is below the maxpool floor of {c_mp['cycles']} cycles/frame (not foldable, ~1.25 cycles per input pixel): "
                             f"this is the minimum frame budget of the whole network")
        pe_pool, pool_first, pool_last = cin, "MaxPool", "MaxPool"
    else:
        sp = search_swg_pool(g_mp, A, F)       # Pool_hls folded over channels; floor is 1 cycle per input pixel
        pe_pool, pool_first, pool_last = sp["pe"], "SWG_p", "Pool"
    g_c = _geom("conv", cin, ccv, H, W, Ho, Wo, k=3, s=2, d=1, p=1)
    g_in = _geom("thr_in", cin, cin, H, W, H, W, op="Thresholding")
    g_dup = _geom("dup", cin, cin, H, W, H, W, op="Dup")
    g_tm = _geom("thr_m", cin, cin, H, W, H, W, op="Thresholding")
    g_cat = _geom("cat", cout, cout, Ho, Wo, Ho, Wo, op="Concat")
    g_act = _geom("thr_act", cout, cout, Ho, Wo, Ho, Wo, op="Thresholding")

    pe_c, simd_c, c_cv = _search_mvau(g_c, bits, F)
    tpe_c = _min_pe(ccv, px_out, F, "Thr_c")
    c_cv = _retarget_threshold(c_cv, g_c, A, tpe_c)
    pe_ti = _min_pe(cin, px_in, F, "Thr_in")
    pe_d = _min_pe(cin, px_in, F, "Dup")
    pe_tm = _min_pe(cin, px_in, F, "Thr_m")
    pe_ta = _min_pe(cout, px_out, F, "Thr_act")
    c_ti = fcm.threshold_node_cost(g_in, A, pe_ti)
    c_dup = fcm.stream_node_cost("dup", g_dup, pe_d)
    c_tm = fcm.threshold_node_cost(g_tm, A, pe_tm)
    c_cat = fcm.stream_node_cost("concat", g_cat)
    c_ta = fcm.threshold_node_cost(g_act, A, pe_ta)
    swu = c_cv["simd_swu"]

    def row(name, op, pe, simd, frame_cycles, lut=0.0, bram=0.0, uram=0.0, dsp=0.0, in_w=0, out_w=0):
        return NodeResult(name=name, op=op, pe=pe, simd=simd, cyc_px=frame_cycles / px_out, frame_cycles=int(frame_cycles),
                          lut=lut, bram18=bram, uram=uram, dsp=dsp, in_width_bits=in_w, out_width_bits=out_w)

    nd = res.nodes
    nd.append(row("Thr_in", "Thresholding_rtl (input quant)", pe_ti, 0, c_ti["cycles"], c_ti["total_lut"], c_ti["thr_bram18"], in_w=pe_ti * A, out_w=pe_ti * A))
    nd.append(row("Dup", "DuplicateStreams", pe_d, 0, c_dup["cycles"], c_dup["total_lut"], in_w=pe_d * A, out_w=pe_d * A))
    nd.append(row("FMPad", "FMPadding 3x3", 0, swu, c_cv["fmpad_cycles"], in_w=swu * A, out_w=swu * A))
    nd.append(row("SWG", "ConvolutionInputGenerator_rtl 3x3 s2", 0, swu, c_cv["swu_cycles"], c_cv["swu_lut"], c_cv["swu_bram18"], c_cv["swu_uram18"],
                  in_w=swu * A, out_w=swu * A))
    nd.append(row("MVAU_c", "MVAU rtl 3x3 s2", pe_c, simd_c, c_cv["mvu_cycles"], c_cv["mvu_lut"], c_cv["wm_bram18"], c_cv["wm_uram18"], c_cv["mvu_dsp"],
                  in_w=simd_c * A, out_w=pe_c * c_cv["acc_bits"]))
    nd.append(row("Thr_c", "Thresholding_rtl (branch quant)", tpe_c, 0, px_out * (ccv // tpe_c), c_cv["thr_lut"], c_cv["thr_bram18"],
                  in_w=tpe_c * c_cv["acc_bits"], out_w=tpe_c * A))
    nd.append(row("Thr_m", "Thresholding_rtl (branch quant, upstream of the pool)", pe_tm, 0, c_tm["cycles"], c_tm["total_lut"], c_tm["thr_bram18"],
                  in_w=pe_tm * A, out_w=pe_tm * A))
    if pool_impl == "streaming":
        nd.append(row("MaxPool", "StreamingMaxPool_hls", 0, 0, c_mp["cycles"], c_mp["total_lut"], c_mp["swu_bram18"], in_w=cin * A, out_w=cin * A))
    else:
        nd.append(row("SWG_p", "ConvolutionInputGenerator depthwise 2x2 s2", 0, sp["pe"], sp["swg_cycles"], sp["swg_lut"], sp["swg_bram18"],
                      sp["swg_uram18"], in_w=sp["pe"] * A, out_w=sp["pe"] * A))
        nd.append(row("Pool", "Pool_hls", sp["pe"], 0, sp["pool_cycles"], sp["pool_lut"], in_w=sp["pe"] * A, out_w=sp["pe"] * A))
    nd.append(row("Concat", "StreamingConcat_hls", 0, 0, c_cat["cycles"], c_cat["total_lut"], in_w=cout * A, out_w=cout * A))
    nd.append(row("Thr_act", "Thresholding_rtl (BN+ReLU)", pe_ta, 0, c_ta["cycles"], c_ta["total_lut"], c_ta["thr_bram18"], in_w=pe_ta * A, out_w=pe_ta * A))

    dw = res.dwcs
    _dwc("Thr_in->Dup", pe_ti * A, pe_d * A, cin, pe_ti, pe_d, px_in, px_out, dw)
    _dwc("Dup->FMPad", pe_d * A, swu * A, cin, pe_d, swu, px_in, px_out, dw)
    _dwc("Dup->Thr_m", pe_d * A, pe_tm * A, cin, pe_d, pe_tm, px_in, px_out, dw)
    _dwc(f"Thr_m->{pool_first}", pe_tm * A, pe_pool * A, cin, pe_tm, pe_pool, px_in, px_out, dw)
    _dwc("MVAU_c->Thr_c", pe_c * c_cv["acc_bits"], tpe_c * c_cv["acc_bits"], ccv, pe_c, tpe_c, px_out, px_out, dw)
    _dwc("FIFOmain->Concat", tpe_c * A, ccv * A, ccv, tpe_c, ccv, px_out, px_out, dw)
    _dwc("skipFIFO->Concat", pe_pool * A, cin * A, cin, pe_pool, cin, px_out, px_out, dw)
    _dwc("Concat->Thr_act", cout * A, pe_ta * A, cout, cout, pe_ta, px_out, px_out, dw)
    for d in dw:
        if d.cyc_px > T + 1e-9:
            res.warnings.append(f"DWC {d.edge} needs {d.cyc_px:.2f} cyc/px (output pixels) > T_out={T:.2f}")

    # rough join-FIFO / latency estimate (the simulation sets the real depths): both branches wait for the same first 2 input rows
    t_first = (W + 2) * T_in
    res.latency_first_out_cycles = int(math.ceil(t_first + nd[4].cyc_px + ccv / tpe_c + 4))
    res.frame_cycles = int(res.latency_first_out_cycles + (px_out - 1) * T)
    w_skip, w_main = cin // pe_pool, ccv // tpe_c
    res.skip_fifo = SkipFifo(width_bits=pe_pool * A, depth_words=4 * w_skip, bits=pe_pool * A * 4 * w_skip,
                             bram18_if_block=_fifo_bram18(pe_pool * A, 4 * w_skip),
                             lutram_luts_if_distributed=math.ceil(pe_pool * A * 4 * w_skip / 64), pixels_buffered=4, pe=pe_pool)
    res.params["main_fifo_words"] = 4 * w_main
    dwc_lut = sum(d.lut for d in dw)
    res.totals = dict(lut=sum(x.lut for x in nd) + dwc_lut, bram18=sum(x.bram18 for x in nd) + res.skip_fifo.bram18_if_block,
                      uram=sum(x.uram for x in nd), dsp=sum(x.dsp for x in nd), dwc_lut=dwc_lut)
    res.balance = dict(max_cyc_px=nd[4].cyc_px, min_cyc_px=nd[4].cyc_px, min_over_max=1.0, mean_util=nd[4].cyc_px / T, scope="MVAU_c")
    if max(x.frame_cycles for x in nd) > F:
        res.warnings.append(f"slowest node {max(x.frame_cycles for x in nd)} cycles/frame exceeds F={F}")
    if pool_impl == "streaming":
        res.warnings.append(f"maxpool floor: F >= {c_mp['cycles']} cycles/frame for this input size; MaxPool runs at {c_mp['cycles'] / F:.0%} of F. "
                            f"Concat / MaxPool LUTs are provisional in finn_cost_model")
    else:
        res.warnings.append(f"Pool route: Pool PE={sp['pe']} runs at {sp['pool_cycles'] / F:.0%} of F (floor 1 cycle per input pixel = {px_in}); "
                            f"Pool LUT is provisional (comparators only)")
    return res


def verify_with_sim(r: BottleneckResult, fifo_depth: int = 2, tol: float = 0.02, max_tries: int = 6, shrink: bool = True,
                    tol_soft: float = 0.03, fifo_mem: str = "auto") -> dict:
    """Paced 2-frame run measures what both join FIFOs need; a saturated 3-frame run escalates the FIFOs in front of / after FMPad
    (next-frame prefetch, see bottleneck.verify_with_sim) until the last frame's period per OUTPUT pixel is within tol of T_out."""
    from int_bottleneck_sim import UNBOUNDED, simulate_int

    p = r.params
    T = p["T"]
    n = {x.name: x for x in r.nodes}
    cf = p["cin"] // n["SWG"].simd
    w_skip, w_main = p["cin"] // r.skip_fifo.pe, p["cmid"] // n["Thr_c"].pe
    emap0 = {"FMPad": (p["width"] + 2) * cf + 2}
    pad_out = (p["width"] + 2) * cf + 2
    paced = simulate_int(r, inject_interval=p["T_in"], skip_depth=UNBOUNDED, main_depth=UNBOUNDED, fifo_depth=fifo_depth, elastic_map=emap0, frames=2)
    need_skip, need_main = paced.fifo_max["skip FIFO"], paced.fifo_max["FIFO main"]
    d = need_skip + w_skip
    f = r.skip_fifo
    f.depth_words, f.bits, f.pixels_buffered = d, f.width_bits * d, math.ceil(d / w_skip)
    f.bram18_if_block, f.lutram_luts_if_distributed = _fifo_bram18(f.width_bits, d), math.ceil(f.width_bits * d / 64)
    p["main_fifo_words"] = max(2, need_main + w_main)
    sat, tries, emap = None, 0, dict(emap0)
    for tries, (es, with_pad, um) in enumerate(_SIZING_SCHEDULE[:max_tries], 1):
        depth = fifo_depth * um
        emap = {k: max(v * es, depth) for k, v in emap0.items()}
        extra = {"FMPad->out": pad_out * es} if with_pad else {}
        sat = simulate_int(r, inject_interval=0, skip_depth=r.skip_fifo.depth_words, main_depth=p["main_fifo_words"], fifo_depth=depth,
                           elastic_map=emap, fifo_depths=extra, frames=3)
        if not sat.deadlock and sat.steady_cyc_px <= T * (1 + tol):
            break
    ok = (not sat.deadlock) and sat.steady_cyc_px <= T * (1 + tol)
    if ok and shrink:
        sized = {k: max(fifo_depth, occ) for k, occ in sat.fifo_max.items() if k not in ("skip FIFO", "FIFO main")}
        small = simulate_int(r, inject_interval=0, skip_depth=r.skip_fifo.depth_words, main_depth=p["main_fifo_words"], fifo_depth=depth,
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
        elastic_depth=emap.get("FMPad"), frame_periods=sat.frame_periods,
    )
    finalize_fifo_costs(r, fifo_mem)
    if not ok:
        raise RuntimeError(f"simulation does not reach T_out={T:.2f}: steady {sat.steady_cyc_px:.2f} cyc/px, deadlock={sat.deadlock}")
    return r.verification


def rate_report(r: BottleneckResult, with_sim: bool = True) -> str:
    from int_bottleneck_sim import simulate_int

    T = r.params["T"]
    n = {x.name: x for x in r.nodes}
    fr = {}
    if with_sim:
        if not r.verification:
            verify_with_sim(r)
        depths = {k: f["depth"] for k, f in r.fifo_graph["fifos"].items()}
        sim = simulate_int(r, inject_interval=0, skip_depth=r.skip_fifo.depth_words, main_depth=r.params["main_fifo_words"], fifo_depth=2, fifo_depths=depths)
        fr = {name: sim.fractions(name, sim.steady_window) for name in sim.node_names if name in n}
    pool_nodes = ["MaxPool"] if r.params.get("pool_impl", "streaming") == "streaming" else ["SWG_p", "Pool"]
    pool_slow = max(pool_nodes, key=lambda x: n[x].cyc_px)
    br = (("shared input", ["Thr_in", "Dup"]), ("CONV branch", ["FMPad", "SWG", "MVAU_c", "Thr_c"]), ("POOL branch", ["Thr_m"] + pool_nodes),
          ("JOIN", ["Concat", "Thr_act"]))
    lines = [f"rate mismatch, budget T_out = {T:.2f} cyc/output pixel (F = {r.params['F']} cycles/frame)"]
    hdr = f"{'node':8s} {'cyc/px':>7s} {'util':>5s}" + ("   busy starv block" if fr else "")
    for title, names in br:
        lines += ["", f"-- {title}", hdr]
        for name in names:
            x = n[name]
            row = f"{name:8s} {x.cyc_px:7.2f} {x.cyc_px / T:5.2f}"
            if fr:
                f = fr[name]
                row += f"   {f['busy']:4.2f}  {f['starved']:4.2f}  {f['blocked']:4.2f}"
            lines.append(row)
    lines += ["", "-- mismatch summary",
              f"conv branch slowest {max(['FMPad', 'SWG', 'MVAU_c', 'Thr_c'], key=lambda x: n[x].cyc_px)} at {max(n[x].cyc_px for x in ['FMPad', 'SWG', 'MVAU_c', 'Thr_c']):.2f}; "
              f"pool branch slowest {pool_slow} at {n[pool_slow].cyc_px:.2f} cyc/px (= {n[pool_slow].frame_cycles / r.params['F']:.0%} of F); budget {T:.2f}",
              f"join FIFOs: skip FIFO (pool end) {r.skip_fifo.depth_words} words x {r.skip_fifo.width_bits} bit, FIFO main {r.params['main_fifo_words']} words"]
    return "\n".join(lines)


_ROLE_OF = {"Thr_in": "thr_in", "Dup": "dup", "FMPad": "fmpad", "SWG": "swg", "MVAU_c": "mvau_c", "Thr_c": "thr_c", "MaxPool": "maxpool", "SWG_p": "swg_p", "Pool": "pool",
            "Thr_m": "thr_m", "Concat": "concat", "Thr_act": "thr_act"}


def to_folding_config(r: BottleneckResult) -> dict:
    """Role-keyed FINN nodeattrs + verified FIFO depths per edge + prediction (schema of bottleneck.to_folding_config). Needs verify_with_sim."""
    if not r.fifo_graph:
        raise RuntimeError("run verify_with_sim(result) before to_folding_config")
    p = r.params
    n = {x.name: x for x in r.nodes}
    block = 1024
    fold = {
        "thr_in": {"PE": n["Thr_in"].pe, "depth_trigger_bram": block}, "dup": {"PE": n["Dup"].pe},
        "fmpad": {"SIMD": n["SWG"].simd}, "swg": {"SIMD": n["SWG"].simd, "parallel_window": int(n["MVAU_c"].simd > p["cin"])},
        "mvau_c": {"PE": n["MVAU_c"].pe, "SIMD": n["MVAU_c"].simd}, "thr_c": {"PE": n["Thr_c"].pe, "depth_trigger_bram": block},
        **({"maxpool": {}} if p.get("pool_impl", "streaming") == "streaming" else
           {"swg_p": {"SIMD": n["SWG_p"].simd, "parallel_window": 0}, "pool": {"PE": n["Pool"].pe}}),
        "thr_m": {"PE": n["Thr_m"].pe, "depth_trigger_bram": block}, "concat": {},
        "thr_act": {"PE": n["Thr_act"].pe, "depth_trigger_bram": block},
    }
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


_OP_LABEL = {"Dup": "DuplicateStreams_hls", "FMPad": "FMPadding_rtl", "SWG": "ConvolutionInputGenerator_rtl", "MVAU_c": "MVAU_rtl",
             "MaxPool": "StreamingMaxPool_hls", "SWG_p": "ConvolutionInputGenerator_rtl", "Pool": "Pool_hls", "Concat": "StreamingConcat_hls"}


def export_onnx(r: BottleneckResult, path: str) -> None:
    """Verified initial-block dataflow graph for Netron (FINN op names, PE/SIMD/cycles/resources as attributes, DWCs, FIFOs with memory mapping)."""
    import onnx
    from onnx import TensorProto, helper

    if not r.fifo_graph:
        raise RuntimeError("run verify_with_sim(result) before export_onnx")
    p, g = r.params, r.fifo_graph
    H, W, Ho, Wo = p["height"], p["width"], p["hout"], p["wout"]
    cin, cout, ccv = p["cin"], p["cout"], p["cmid"]
    by_name = {x.name: x for x in r.nodes}
    slowest = max(x.frame_cycles for x in r.nodes)
    node_shape = {"Thr_in": (cin, H, W), "Dup": (cin, H, W), "FMPad": (cin, H + 2, W + 2), "SWG": (9 * cin, Ho, Wo), "MVAU_c": (ccv, Ho, Wo),
                  "Thr_c": (ccv, Ho, Wo), "MaxPool": (cin, Ho, Wo), "SWG_p": (4 * cin, Ho, Wo), "Pool": (cin, Ho, Wo), "Thr_m": (cin, H, W), "Concat": (cout, Ho, Wo), "Thr_act": (cout, Ho, Wo)}
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
            attrs = dict(stage="init_block", shape_CHW="x".join(map(str, shp)), pe=x.pe, simd=x.simd, cycles=int(x.frame_cycles),
                         ii_cycles_per_output_pixel=float(x.cyc_px), pct_of_budget=100 * x.frame_cycles / p["F"],
                         is_slowest_node=int(x.frame_cycles == slowest), lut=float(x.lut), bram18k=float(x.bram18), uram18=float(x.uram), dsp=int(x.dsp),
                         in_width_bits=x.in_width_bits, out_width_bits=x.out_width_bits, act_bits=p["bits"])
            op = _OP_LABEL.get(name, "Thresholding_rtl")
        else:
            in_bits, out_bits = g["fifos"][ins[0]]["bits"], g["fifos"][outs[0]]["bits"]
            shp = shape[in_t[0]]
            attrs = dict(stage="init_block", shape_CHW="x".join(map(str, shp)), inWidth=in_bits, outWidth=out_bits,
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
                "StreamingFIFO_rtl", [tin], [f"{fname}:out"], name=fname, domain="finn_milp", stage="init_block", shape_CHW="x".join(map(str, shape[tin])),
                depth=int(depth), width_bits=int(bits), bits=int(depth * bits), max_occupancy=int(f.get("max_occ", 0)),
                is_skip_fifo=int(fname == "skip FIFO"), is_join_fifo=int(fname in ("skip FIFO", "FIFO main")),
                producer=f["producer"], consumer=f["consumer"], **_fifo_attrs(r, fname)))
            shape[f"{fname}:out"] = shape[tin]
            wired.append(f"{fname}:out")
        out_names = [out_tensor(name, o) for o in outs]
        onnx_nodes.append(helper.make_node(op, wired, out_names, name=name, domain="finn_milp", **attrs))
        for t in out_names:
            shape[t] = shp
    final_out = out_tensor("Thr_act", g["io"]["Thr_act"][1][0])
    info = lambda t: helper.make_tensor_value_info(t, TensorProto.FLOAT, [1, *shape[t]])
    value_infos = [info(t) for t in shape if t not in ("global_in", final_out)]
    v = r.verification
    summary = (f"init block Cin={cin} Cout={cout} {H}x{W}->{Ho}x{Wo} INT{p['bits']} T_out={p['T']:.2f} F={p['F']} | LUT {r.totals['lut']:.0f} "
               f"BRAM18 {r.totals['bram18']:.1f} URAM {r.totals['uram']:.0f} DSP {r.totals['dsp']:.0f} | sim steady {v.get('steady_cyc_px', float('nan')):.2f} "
               f"cyc/px, latency {v.get('latency_first_out')} cycles")
    graph = helper.make_graph(onnx_nodes, "init_block_dataflow", [info("global_in")], [info(final_out)], value_info=value_infos, doc_string=summary)
    model = helper.make_model(graph, producer_name="int_bottleneck_analytical",
                              opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("finn_milp", 1)])
    onnx.save(model, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("cin", "cout", "bits", "height", "width"):
        ap.add_argument(f"--{a}", type=int, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--F", type=int, help="frame cycles budget")
    g.add_argument("--T-out", type=float, help="cycles per OUTPUT pixel")
    ap.add_argument("--pool-impl", choices=("streaming", "swg_pool"), default="streaming")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--rates", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--onnx", metavar="PATH")
    ap.add_argument("--folding-json", metavar="PATH")
    a = ap.parse_args()
    r = model_int_bottleneck(a.cin, a.cout, a.bits, a.height, a.width, a.F, a.T_out, a.pool_impl)
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
