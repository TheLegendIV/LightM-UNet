"""Analytical model of one ENet residual bottleneck on FINN.

    Dup -> MVAU_r -> Thr_r -> FMPad -> SWG_m -> MVAU_m -> Thr_m -> MVAU_e -> Thr_e -> Add -> Thr_out
      |                                                                                  |
      +-------------------------------> Thr_s -> [skip FIFO] ----------------------------+

Conventions (see analytical.md): rate = cycles per output pixel (cyc/px), lower = faster.
T is the target cyc/px of the whole bottleneck. Every node is folded to be as close to
T as its divisor constraints allow, from below (cycles <= T, maximise cycles), so the
chain is as rate balanced as possible and no node idles more than it has to.
DWCs are allowed on every edge where adjacent stream widths differ, and are priced.

Resource numbers come from finn_cost_model.py (force_dsp on, ram_style auto, RTL MVAU,
standalone Thresholding_rtl), see finn_cost_model.md. Latency and skip FIFO depth are
first-order estimates (pipeline fill of the sliding window dominates).

Run: python3 bottleneck.py --cin 32 --k 3 --dilation 8 --stride 1 --v 4 --z 4 --T 72 --bits 4 --height 32 --width 32
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import finn_cost_model as fcm  # noqa: E402
from finn_cost_model import LayerGeometry  # noqa: E402


# ---------------------------------------------------------------- results

@dataclass
class NodeResult:
    name: str
    op: str
    pe: int
    simd: int            # 0 where not applicable
    cyc_px: float        # cycles per bottleneck output pixel
    frame_cycles: int
    lut: float = 0.0
    bram18: float = 0.0
    uram: float = 0.0
    dsp: float = 0.0
    out_width_bits: int = 0   # stream width this node emits
    in_width_bits: int = 0    # stream width this node consumes


@dataclass
class DwcResult:
    edge: str
    in_width: int
    out_width: int
    lut: float
    cyc_px: float


@dataclass
class SkipFifo:
    width_bits: int
    depth_words: int
    bits: int
    bram18_if_block: int
    lutram_luts_if_distributed: int
    pixels_buffered: int
    pe: int


@dataclass
class BottleneckResult:
    params: dict
    nodes: list = field(default_factory=list)
    dwcs: list = field(default_factory=list)
    skip_fifo: SkipFifo | None = None
    totals: dict = field(default_factory=dict)
    latency_first_out_cycles: int = 0
    frame_cycles: int = 0
    balance: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    verification: dict = field(default_factory=dict)   # filled by verify_with_sim
    fifo_graph: dict = field(default_factory=dict)     # simulated topology + verified FIFO depths

    def to_dict(self) -> dict:
        return asdict(self)

    def report(self) -> str:
        lines = []
        p = self.params
        lines.append(
            f"bottleneck: Cin={p['cin']} Cmid={p['cmid']} Cout={p['cout']} k={p['k']} d={p['dilation']} s={p['stride']} "
            f"{p['height']}x{p['width']} INT{p['bits']}  target T={p['T']} cyc/px"
        )
        lines.append(f"{'node':10s} {'op':26s} {'PE':>4s} {'SIMD':>5s} {'cyc/px':>8s} {'util':>6s} {'LUT':>9s} {'BRAM18':>7s} {'URAM':>5s} {'DSP':>5s}")
        for n in self.nodes:
            util = n.cyc_px / p["T"]
            lines.append(
                f"{n.name:10s} {n.op:26s} {n.pe:4d} {n.simd:5d} {n.cyc_px:8.2f} {util:6.2f} {n.lut:9.0f} {n.bram18:7.1f} {n.uram:5.0f} {n.dsp:5.0f}"
            )
        if self.dwcs:
            lines.append("DWCs:")
            for d in self.dwcs:
                lines.append(f"  {d.edge:22s} {d.in_width:5d} -> {d.out_width:5d} bits  LUT {d.lut:7.0f}  {d.cyc_px:6.2f} cyc/px")
        f = self.skip_fifo
        lines.append(
            f"skip FIFO: width {f.width_bits} bits (PE_ts={f.pe}), depth {f.depth_words} words "
            f"({f.pixels_buffered} pixels), {f.bits} bits, ~{f.bram18_if_block} BRAM18 if block / ~{f.lutram_luts_if_distributed} LUT if distributed"
        )
        t = self.totals
        lines.append(f"totals: LUT {t['lut']:.0f}  BRAM18 {t['bram18']:.1f}  URAM {t['uram']:.0f}  DSP {t['dsp']:.0f}  (DWC LUT {t['dwc_lut']:.0f})")
        lines.append(f"latency (first pixel in -> first pixel out): {self.latency_first_out_cycles} cycles; frame ~ {self.frame_cycles} cycles")
        b = self.balance
        lines.append(f"balance ({b['scope']}): slowest {b['max_cyc_px']:.2f}, fastest {b['min_cyc_px']:.2f} cyc/px, min/max = {b['min_over_max']:.2f}, mean util = {b['mean_util']:.2f}")
        v = self.verification
        if v:
            lines.append(
                f"simulation: {'PASS' if v['ok'] else 'FAIL'}  steady {v['steady_cyc_px']:.2f} cyc/px (T={p['T']}), "
                f"first-out latency {v['latency_first_out']} cycles, skip FIFO needs {v['skip_needed_words']} words, "
                f"elastic FIFO before FMPad {v['elastic_depth']} words ({v['tries']} sizing tries)"
            )
        for w in self.warnings:
            lines.append("WARNING: " + w)
        return "\n".join(lines)


# ---------------------------------------------------------------- helpers

def _geom(name, cin, cout, hin, win, hout, wout, k=1, s=1, d=1, p=0, op="Conv2d") -> LayerGeometry:
    return LayerGeometry(
        op_type=op, name=name, stage="bottleneck", cin=cin, hin=hin, win=win, cout=cout, hout=hout, wout=wout,
        kh=k, kw=k, sh=s, sw=s, dh=d, dw=d, groups=1, ph=p, pw=p,
    )


def _min_pe(channels: int, pixels: int, budget: int, what: str) -> int:
    """Smallest PE | channels with pixels*channels/PE <= budget (narrowest = cheapest, closest to the target)."""
    for pe in fcm.divisors(channels):
        if pixels * (channels // pe) <= budget:
            return pe
    raise ValueError(f"{what}: even PE={channels} needs {pixels} cycles per frame > budget {budget}")


def _simd_candidates(cin: int, k: int) -> list[int]:
    """SWG SIMD = MVAU SIMD must divide Cin; Cin*j (j | k*k) is the parallel-window mode."""
    cands = set(fcm.divisors(cin))
    for j in fcm.divisors(k * k):
        cands.add(cin * j)
    return sorted(cands)


def _search_mvau(layer: LayerGeometry, bits: int, budget: int, weight_bits: int | None = None) -> tuple[int, int, dict]:
    """(PE, SIMD, cost) with the largest cycles <= budget (fully balanced = cycles == budget);
    ties broken by BRAM then LUT. weight_bits defaults to bits."""
    wb = bits if weight_bits is None else weight_bits
    best = None
    for pe in fcm.divisors(layer.cout):
        for simd in _simd_candidates(layer.cin, layer.kh):
            cost = fcm.conv_cost_pe_simd(layer, wb, bits, pe, simd, ram_style=fcm.RAM_STYLE_AUTO, force_dsp=True)
            if cost["mvu_cycles"] > budget or cost["fmpad_cycles"] > budget:
                continue
            key = (-cost["mvu_cycles"], cost["wm_bram18"] + cost["swu_bram18"] + cost["thr_bram18"], cost["total_lut"])
            if best is None or key < best[0]:
                best = (key, pe, simd)
    if best is None:
        raise ValueError(
            f"{layer.name}: no (PE, SIMD) reaches {budget} cycles/frame; fastest is "
            f"{layer.hout * layer.wout} cycles/frame (PE=Cout, SIMD=MW). Raise T."
        )
    _, pe, simd = best
    cost = fcm.layer_cost_pe_simd_auto_ram(layer, wb, bits, pe, simd, force_dsp=True)
    return pe, simd, cost


def _retarget_threshold(cost: dict, layer: LayerGeometry, bits: int, thr_pe: int) -> dict:
    """Replace the cost model's built-in standalone-threshold terms by one at an explicit PE_t."""
    lut, bram, uram = fcm._thresholding_rtl_cost(thr_pe, bits, layer.cout, ram_style="block", in_bits=cost["acc_bits"])
    out = dict(cost)
    out["total_lut"] = cost["total_lut"] - cost["thr_lut"] + lut
    out["thr_lut"], out["thr_bram18"], out["thr_uram18"], out["thr_pe"] = lut, bram, uram, thr_pe
    return out


_BRAM18_SDP_ASPECTS = ((1, 16384), (2, 8192), (4, 4096), (9, 2048), (18, 1024), (36, 512))  # UG573 table 1-10


def _fifo_bram18(width: int, depth: int) -> int:
    """BRAM18 count of a (width x depth) FIFO memory over the full SDP aspect-ratio table.
    A narrow, deep memory (the skip FIFO after Thr_s with small PE_ts) packs far better than a wide one."""
    return min(math.ceil(width / pw) * math.ceil(depth / pd) for pw, pd in _BRAM18_SDP_ASPECTS)


def _dwc(edge: str, in_w: int, out_w: int, channels: int, pe_in: int, pe_out: int, pixels: int, px_ref: int, out: list) -> None:
    if in_w == out_w:
        return
    cost = fcm.dwc_cost(in_w, out_w)
    cyc_px = channels / min(pe_in, pe_out) * pixels / px_ref
    out.append(DwcResult(edge, in_w, out_w, cost["total_lut"], cyc_px))


# ---------------------------------------------------------------- model

def model_bottleneck(
    cin: int, v: int, z: int, T: int, bits: int, height: int, width: int,
    k: int = 3, dilation: int = 1, stride: int = 1,
) -> BottleneckResult:
    """cin: input channels; Cmid = cin/v; Cout = Cmid*z. T: target cyc/px (per bottleneck output pixel).
    bits: uniform weight and activation bits. height/width: feature map size at the bottleneck input."""
    if stride != 1:
        raise ValueError("stride != 1: the identity skip then needs a subsampling path with a different pixel count; not modeled.")
    if cin % v:
        raise ValueError(f"Cin={cin} not divisible by v={v}")
    cmid = cin // v
    cout = cmid * z
    if cout != cin:
        raise ValueError(f"identity skip needs Cout == Cin (v == z); got Cin={cin}, Cout={cout}")
    k_eff = (k - 1) * dilation + 1
    pad = (k_eff - 1) // 2  # same padding
    hout = (height + 2 * pad - k_eff) // stride + 1
    wout = (width + 2 * pad - k_eff) // stride + 1
    px = hout * wout
    budget = T * px
    A = bits

    res = BottleneckResult(params=dict(
        cin=cin, cmid=cmid, cout=cout, v=v, z=z, T=T, bits=bits, k=k, dilation=dilation, stride=stride,
        height=height, width=width, hout=hout, wout=wout, pad=pad,
    ))

    g_r = _geom("reduce", cin, cmid, height, width, height, width)
    g_m = _geom("mid", cmid, cmid, height, width, hout, wout, k=k, s=stride, d=dilation, p=pad)
    g_e = _geom("expand", cmid, cout, hout, wout, hout, wout)

    # --- MVAUs: largest cycles <= budget
    pe_r, simd_r, c_r = _search_mvau(g_r, bits, budget)
    pe_m, simd_m, c_m = _search_mvau(g_m, bits, budget)
    pe_e, simd_e, c_e = _search_mvau(g_e, bits, budget)

    # --- thresholds inside the convs: narrowest PE_t that fits the budget (rule 2, DWC allowed)
    tpe_r = _min_pe(cmid, px, budget, "Thr_r")
    tpe_m = _min_pe(cmid, px, budget, "Thr_m")
    tpe_e = _min_pe(cout, px, budget, "Thr_e")
    c_r = _retarget_threshold(c_r, g_r, A, tpe_r)
    c_m = _retarget_threshold(c_m, g_m, A, tpe_m)
    c_e = _retarget_threshold(c_e, g_e, A, tpe_e)

    # --- stream nodes and join thresholds
    g_dup = _geom("dup", cin, cin, height, width, height, width, op="Dup")
    g_join = _geom("join", cout, cout, hout, wout, hout, wout, op="Thresholding")
    pe_d = _min_pe(cin, px, budget, "Dup")
    pe_ts = _min_pe(cout, px, budget, "Thr_s")
    pe_a = _min_pe(cout, px, budget, "Add")
    pe_to = _min_pe(cout, px, budget, "Thr_out")
    c_dup = fcm.stream_node_cost("dup", g_dup, pe_d)
    c_add = fcm.stream_node_cost("add", g_join, pe_a)
    c_ts = fcm.threshold_node_cost(g_join, A, pe_ts)
    c_to = fcm.threshold_node_cost(g_join, A, pe_to)

    acc_r, acc_m, acc_e = c_r["acc_bits"], c_m["acc_bits"], c_e["acc_bits"]
    add_bits = A + 1

    def node(name, op, pe, simd, cyc_frame, cost, in_w, out_w, lut=None):
        return NodeResult(
            name=name, op=op, pe=pe, simd=simd, cyc_px=cyc_frame / px, frame_cycles=int(cyc_frame),
            lut=cost["total_lut"] if lut is None else lut,
            bram18=cost.get("swu_bram18", 0) + cost.get("wm_bram18", 0) + cost.get("thr_bram18", 0),
            uram=cost.get("swu_uram18", 0) + cost.get("wm_uram18", 0) + cost.get("thr_uram18", 0),
            dsp=cost.get("total_dsp", 0), in_width_bits=in_w, out_width_bits=out_w,
        )

    nodes = res.nodes
    nodes.append(node("Dup", "DuplicateStreams", pe_d, 0, c_dup["cycles"], c_dup, pe_d * A, pe_d * A))
    # reduce: MVAU row (without its threshold) + Thr_r row
    nodes.append(NodeResult("MVAU_r", "MVAU rtl 1x1", pe_r, simd_r, c_r["mvu_cycles"] / px, c_r["mvu_cycles"],
                            lut=c_r["mvu_lut"], bram18=c_r["wm_bram18"], uram=c_r["wm_uram18"], dsp=c_r["mvu_dsp"],
                            in_width_bits=simd_r * A, out_width_bits=pe_r * acc_r))
    thr_cyc = lambda c, pe_t: px * (c // pe_t)
    nodes.append(NodeResult("Thr_r", "Thresholding_rtl", tpe_r, 0, thr_cyc(cmid, tpe_r) / px, thr_cyc(cmid, tpe_r),
                            lut=c_r["thr_lut"], bram18=c_r["thr_bram18"], in_width_bits=tpe_r * acc_r, out_width_bits=tpe_r * A))
    simd_swu = c_m["simd_swu"]
    if c_m["fmpad_cycles"]:
        nodes.append(NodeResult("FMPad", "FMPadding", 0, simd_swu, c_m["fmpad_cycles"] / px, c_m["fmpad_cycles"],
                                in_width_bits=simd_swu * A, out_width_bits=simd_swu * A))
    nodes.append(NodeResult("SWG_m", "ConvolutionInputGenerator_rtl", 0, simd_swu, c_m["swu_cycles"] / px, c_m["swu_cycles"],
                            lut=c_m["swu_lut"], bram18=c_m["swu_bram18"], uram=c_m["swu_uram18"],
                            in_width_bits=simd_swu * A, out_width_bits=simd_swu * A))
    nodes.append(NodeResult("MVAU_m", f"MVAU rtl {k}x{k} d={dilation}", pe_m, simd_m, c_m["mvu_cycles"] / px, c_m["mvu_cycles"],
                            lut=c_m["mvu_lut"], bram18=c_m["wm_bram18"], uram=c_m["wm_uram18"], dsp=c_m["mvu_dsp"],
                            in_width_bits=simd_m * A, out_width_bits=pe_m * acc_m))
    nodes.append(NodeResult("Thr_m", "Thresholding_rtl", tpe_m, 0, thr_cyc(cmid, tpe_m) / px, thr_cyc(cmid, tpe_m),
                            lut=c_m["thr_lut"], bram18=c_m["thr_bram18"], in_width_bits=tpe_m * acc_m, out_width_bits=tpe_m * A))
    nodes.append(NodeResult("MVAU_e", "MVAU rtl 1x1", pe_e, simd_e, c_e["mvu_cycles"] / px, c_e["mvu_cycles"],
                            lut=c_e["mvu_lut"], bram18=c_e["wm_bram18"], uram=c_e["wm_uram18"], dsp=c_e["mvu_dsp"],
                            in_width_bits=simd_e * A, out_width_bits=pe_e * acc_e))
    nodes.append(NodeResult("Thr_e", "Thresholding_rtl", tpe_e, 0, thr_cyc(cout, tpe_e) / px, thr_cyc(cout, tpe_e),
                            lut=c_e["thr_lut"], bram18=c_e["thr_bram18"], in_width_bits=tpe_e * acc_e, out_width_bits=tpe_e * A))
    nodes.append(node("Thr_s", "Thresholding_rtl (skip)", pe_ts, 0, c_ts["cycles"], c_ts, pe_ts * A, pe_ts * A))
    nodes.append(node("Add", "AddStreams", pe_a, 0, c_add["cycles"], c_add, pe_a * A, pe_a * add_bits))
    nodes.append(node("Thr_out", "Thresholding_rtl (join)", pe_to, 0, c_to["cycles"], c_to, pe_to * add_bits, pe_to * A))

    # --- DWCs on every edge with a width mismatch
    dw = res.dwcs
    _dwc("Dup->MVAU_r", pe_d * A, simd_r * A, cin, pe_d, simd_r, px, px, dw)
    _dwc("MVAU_r->Thr_r", pe_r * acc_r, tpe_r * acc_r, cmid, pe_r, tpe_r, px, px, dw)
    _dwc("Thr_r->SWG_m", tpe_r * A, simd_swu * A, cmid, tpe_r, simd_swu, px, px, dw)
    _dwc("MVAU_m->Thr_m", pe_m * acc_m, tpe_m * acc_m, cmid, pe_m, tpe_m, px, px, dw)
    _dwc("Thr_m->MVAU_e", tpe_m * A, simd_e * A, cmid, tpe_m, simd_e, px, px, dw)
    _dwc("MVAU_e->Thr_e", pe_e * acc_e, tpe_e * acc_e, cout, pe_e, tpe_e, px, px, dw)
    _dwc("Thr_e->Add", tpe_e * A, pe_a * A, cout, tpe_e, pe_a, px, px, dw)
    _dwc("Dup->Thr_s", pe_d * A, pe_ts * A, cout, pe_d, pe_ts, px, px, dw)
    _dwc("FIFO->Add (skip)", pe_ts * A, pe_a * A, cout, pe_ts, pe_a, px, px, dw)  # widen AFTER the FIFO
    _dwc("Add->Thr_out", pe_a * add_bits, pe_to * add_bits, cout, pe_a, pe_to, px, px, dw)
    for d in dw:
        if d.cyc_px > T:
            res.warnings.append(f"DWC {d.edge} needs {d.cyc_px:.1f} cyc/px > T")

    # --- latency (first pixel in -> first pixel out) and skip FIFO
    cyc = {n.name: n.cyc_px for n in nodes}
    dup_px = cin / pe_d
    cf_pad = cmid / simd_swu
    pad_top = pad * (width + 2 * pad) * cf_pad
    n_fill = pad * width + pad + 1                       # real pixels the SWG needs before window (0,0) exists
    t_first = cyc["MVAU_r"] + cmid / tpe_r
    t_window = max(t_first, pad_top) + (n_fill - 1) * T
    swg_px = k * k * cmid / simd_m
    l_main = t_window + swg_px + cyc["MVAU_m"] + cmid / tpe_m + cyc["MVAU_e"] + cout / tpe_e
    skip_px = math.ceil((l_main - cout / pe_ts) / T) + 1
    words_px = cout // pe_ts
    depth = skip_px * words_px
    fifo_w = pe_ts * A
    res.skip_fifo = SkipFifo(
        width_bits=fifo_w, depth_words=depth, bits=fifo_w * depth, bram18_if_block=_fifo_bram18(fifo_w, depth),
        lutram_luts_if_distributed=math.ceil(fifo_w * depth / 64), pixels_buffered=skip_px, pe=pe_ts,
    )
    res.latency_first_out_cycles = int(math.ceil(dup_px + l_main + cout / pe_a + cout / pe_to))
    res.frame_cycles = int(res.latency_first_out_cycles + (px - 1) * T)

    # --- totals
    dwc_lut = sum(d.lut for d in dw)
    res.totals = dict(
        lut=sum(n.lut for n in nodes) + dwc_lut,
        bram18=sum(n.bram18 for n in nodes) + res.skip_fifo.bram18_if_block,
        uram=sum(n.uram for n in nodes), dsp=sum(n.dsp for n in nodes), dwc_lut=dwc_lut,
    )
    mv = [n for n in nodes if n.op.startswith("MVAU")]
    cycs = [n.cyc_px for n in mv]
    res.balance = dict(
        max_cyc_px=max(cycs), min_cyc_px=min(cycs), min_over_max=min(cycs) / max(cycs),
        mean_util=sum(cycs) / len(cycs) / T, scope="MVAU nodes",
    )
    if max(n.cyc_px for n in nodes) > T + 1e-9:
        res.warnings.append(f"slowest node {max(n.cyc_px for n in nodes):.2f} cyc/px exceeds T={T}")
    for n in mv:
        if n.cyc_px < T - 1e-9:
            res.warnings.append(
                f"{n.name} runs at {n.cyc_px:.1f} < T={T} cyc/px ({n.cyc_px / T:.0%}): no (PE, SIMD) divisor pair lands exactly on T")
    return res


# ---------------------------------------------------------------- simulation verification

def verify_with_sim(r: BottleneckResult, fifo_depth: int = 2, tol: float = 0.01, max_tries: int = 8, shrink: bool = True) -> dict:
    """Run bottleneck_sim on the chosen foldings and size the FIFOs until the simulated throughput meets T.

    1. Input paced at T with an unbounded skip FIFO: measures the skip FIFO the design really needs and the latency;
       the skip FIFO is raised to that if the analytic estimate was short.
    2. Saturated input: the elastic FIFO in front of FMPad starts at (pad+1) pixels (the SWG needs pad+1 new real
       pixels at every output row start) and doubles until the steady cyc/px is within tol of T. Without it the
       SWG starves the 3x3 MVAU at each row start.
    The verified FIFO depths are stored in r.fifo_graph for export_onnx. Raises RuntimeError if no depth reaches T."""
    from bottleneck_sim import UNBOUNDED, simulate

    T, pad = r.params["T"], r.params["pad"]
    cf = r.params["cmid"] // next(n for n in r.nodes if n.name == "SWG_m").simd
    words_px = r.params["cout"] // r.skip_fifo.pe

    elastic = (pad + 1) * cf + 2
    paced = simulate(r, inject_interval=T, skip_depth=UNBOUNDED, fifo_depth=fifo_depth, elastic_depth=elastic)
    need = paced.fifo_max["skip FIFO"]
    if need > r.skip_fifo.depth_words:
        d = need + words_px
        r.warnings.append(f"skip FIFO raised from {r.skip_fifo.depth_words} to {d} words by simulation")
        f = r.skip_fifo
        f.depth_words, f.bits, f.pixels_buffered = d, f.width_bits * d, math.ceil(d / words_px)
        f.bram18_if_block, f.lutram_luts_if_distributed = _fifo_bram18(f.width_bits, d), math.ceil(f.width_bits * d / 64)

    # ordinary FIFOs: the upstream chain runs only ~11% faster than T, so it needs a little elasticity everywhere;
    # double one uniform depth until the saturated run reaches T, then shrink every FIFO to its observed occupancy.
    sat = None
    tries = 0
    depth = fifo_depth
    for tries in range(1, max_tries + 1):
        elastic = max(elastic, depth)
        sat = simulate(r, inject_interval=0, skip_depth=r.skip_fifo.depth_words, fifo_depth=depth, elastic_depth=elastic)
        if not sat.deadlock and sat.steady_cyc_px <= T * (1 + tol):
            break
        depth *= 2
        elastic *= 2
    if not sat.deadlock and sat.steady_cyc_px <= T * (1 + tol) and shrink:
        sized = {n: max(fifo_depth, occ) for n, occ in sat.fifo_max.items() if n != "skip FIFO"}
        small = simulate(r, inject_interval=0, skip_depth=r.skip_fifo.depth_words, fifo_depth=depth, elastic_depth=elastic,
                         fifo_depths=sized)
        if not small.deadlock and small.steady_cyc_px <= T * (1 + tol):
            sat = small
    ok = (not sat.deadlock) and sat.steady_cyc_px <= T * (1 + tol)
    graph = sat.graph
    for name, occ in sat.fifo_max.items():
        graph["fifos"][name]["max_occ"] = occ
    r.fifo_graph = graph
    r.verification = dict(
        ok=ok, steady_cyc_px=sat.steady_cyc_px, latency_first_out=paced.latency_first_out, frame_cycles=sat.cycles,
        skip_needed_words=need, elastic_depth=elastic, uniform_depth=depth, tries=tries, deadlock=sat.deadlock,
    )
    if not ok:
        raise RuntimeError(
            f"simulation does not reach T={T}: steady {sat.steady_cyc_px:.2f} cyc/px, deadlock={sat.deadlock}, "
            f"elastic depth tried up to {elastic} words"
        )
    return r.verification


# ---------------------------------------------------------------- ONNX export (same style as MILP/milp_outputs.py)

_OP_LABEL = {
    "Dup": "DuplicateStreams_hls", "MVAU_r": "MVAU_rtl", "MVAU_m": "MVAU_rtl", "MVAU_e": "MVAU_rtl",
    "FMPad": "FMPadding_rtl", "SWG_m": "ConvolutionInputGenerator_rtl", "Add": "AddStreams_hls",
}


def export_onnx(r: BottleneckResult, path: str) -> None:
    """Write the verified dataflow graph as ONNX for Netron: one node per hardware node (FINN op names, PE/SIMD/cycles/
    resources as attributes), a StreamingDataWidthConverter wherever the simulation inserted one, and a StreamingFIFO_rtl
    on every edge (depth, width, simulated max occupancy). Needs verify_with_sim to have run."""
    import onnx
    from onnx import TensorProto, helper

    if not r.fifo_graph:
        raise RuntimeError("run verify_with_sim(result) before export_onnx")
    p, g = r.params, r.fifo_graph
    H, W, pad, k = p["height"], p["width"], p["pad"], p["k"]
    cin, cmid, cout = p["cin"], p["cmid"], p["cout"]
    by_name = {n.name: n for n in r.nodes}
    slowest = max(n.cyc_px for n in r.nodes if n.op.startswith("MVAU"))
    node_shape = {
        "Dup": (cin, H, W), "MVAU_r": (cmid, H, W), "Thr_r": (cmid, H, W), "FMPad": (cmid, H + 2 * pad, W + 2 * pad),
        "SWG_m": (k * k * cmid, H, W), "MVAU_m": (cmid, H, W), "Thr_m": (cmid, H, W), "MVAU_e": (cout, H, W),
        "Thr_e": (cout, H, W), "Thr_s": (cout, H, W), "Add": (cout, H, W), "Thr_out": (cout, H, W),
    }
    ends = {"Source", "Sink"}
    fifos = {n: f for n, f in g["fifos"].items() if f["producer"] not in ends and f["consumer"] not in ends}
    shape = {"global_in": (cin, H, W)}
    onnx_nodes = []

    def out_tensor(node: str, fifo: str) -> str:
        outs = g["io"][node][1]
        return f"{node}:out" if len(outs) == 1 else f"{node}:out{outs.index(fifo)}"

    for name in g["nodes"]:
        if name in ends:
            continue
        ins, outs = g["io"][name]
        in_t = ["global_in" if g["fifos"][f]["producer"] == "Source" else out_tensor(g["fifos"][f]["producer"], f) for f in ins]
        if name in by_name:
            n = by_name[name]
            shp = node_shape[name]
            attrs = dict(
                stage="bottleneck", shape_CHW="x".join(map(str, shp)), pe=n.pe, simd=n.simd, cycles=int(n.frame_cycles),
                ii_cycles_per_pixel=float(n.cyc_px), pct_of_slowest_node=100 * n.cyc_px / slowest,
                is_slowest_node=int(n.op.startswith("MVAU") and abs(n.cyc_px - slowest) < 1e-9),
                lut=float(n.lut), bram18k=float(n.bram18), uram18=float(n.uram), dsp=int(n.dsp),
                in_width_bits=n.in_width_bits, out_width_bits=n.out_width_bits, act_bits=p["bits"],
            )
            op = _OP_LABEL.get(name, "Thresholding_rtl")
        else:  # DWC(prev->next): widths are those of the FIFOs on either side
            in_bits, out_bits = g["fifos"][ins[0]]["bits"], g["fifos"][outs[0]]["bits"]
            shp = shape[in_t[0]]
            attrs = dict(
                stage="bottleneck", shape_CHW="x".join(map(str, shp)), inWidth=in_bits, outWidth=out_bits,
                lut=float(fcm.dwc_cost(in_bits, out_bits)["total_lut"]), bram18k=0.0, uram18=0.0, dsp=0,
            )
            op = "StreamingDataWidthConverter_rtl"
        wired = []
        for fname, tin in zip(ins, in_t):
            if fname not in fifos:
                wired.append(tin)
                continue
            f = fifos[fname]
            depth, bits = f["depth"], f["bits"]
            onnx_nodes.append(helper.make_node(
                "StreamingFIFO_rtl", [tin], [f"{fname}:out"], name=fname, domain="finn_milp",
                stage="bottleneck", shape_CHW="x".join(map(str, shape[tin])), depth=int(depth), width_bits=int(bits),
                bits=int(depth * bits), max_occupancy=int(f.get("max_occ", 0)),
                bram18_if_block=int(_fifo_bram18(bits, depth)), lutram_luts_if_distributed=int(math.ceil(bits * depth / 64)),
                is_skip_fifo=int(fname == "skip FIFO"), producer=f["producer"], consumer=f["consumer"],
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
        f"bottleneck Cin={cin} Cmid={cmid} Cout={cout} k={k} d={p['dilation']} {H}x{W} INT{p['bits']} T={p['T']} | "
        f"LUT {r.totals['lut']:.0f} BRAM18 {r.totals['bram18']:.1f} DSP {r.totals['dsp']:.0f} | "
        f"sim steady {v.get('steady_cyc_px', float('nan')):.2f} cyc/px, latency {v.get('latency_first_out')} cycles"
    )
    graph = helper.make_graph(
        onnx_nodes, "bottleneck_dataflow", [info("global_in")], [info(final_out)], value_info=value_infos, doc_string=summary,
    )
    model = helper.make_model(
        graph, producer_name="bottleneck_analytical",
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("finn_milp", 1)],
    )
    onnx.save(model, path)


# ---------------------------------------------------------------- FINN folding / FIFO config for hardware probes

# role -> FINN-facing role name used by hardware/builds/bottleneck_probe_v1 (stable keys, no FINN node names)
_ROLE_OF = {
    "Dup": "dup", "MVAU_r": "mvau_r", "Thr_r": "thr_r", "FMPad": "fmpad", "SWG_m": "swg", "MVAU_m": "mvau_m",
    "Thr_m": "thr_m", "MVAU_e": "mvau_e", "Thr_e": "thr_e", "Thr_s": "thr_s", "Add": "add", "Thr_out": "thr_out",
}


def to_folding_config(r: BottleneckResult) -> dict:
    """Everything a FINN probe build needs, keyed by role (dup, mvau_r, ..., thr_out), plus the prediction to compare against.

    folding[role]  FINN nodeattrs to apply (PE / SIMD / parallel_window / depth_trigger_bram).
    fifos          one entry per FIFO edge of the verified simulation: producer/consumer role ('dwc' for converters),
                   depth in words, width in bits, simulated max occupancy. The probe build forces these depths.
    predicted      throughput / latency / resources of the model and the simulation (for compare_probe_vs_model).
    Needs verify_with_sim(r) to have run (FIFO depths come from its sized graph)."""
    if not r.fifo_graph:
        raise RuntimeError("run verify_with_sim(result) before to_folding_config")
    p = r.params
    n = {x.name: x for x in r.nodes}
    k, cmid = p["k"], p["cmid"]
    block = 1024  # depth_trigger_bram for ram_style "block" (finn_cost_model.THR_DEPTH_TRIGGER_BRAM)
    folding = {
        "dup": {"PE": n["Dup"].pe},
        "mvau_r": {"PE": n["MVAU_r"].pe, "SIMD": n["MVAU_r"].simd},
        "thr_r": {"PE": n["Thr_r"].pe, "depth_trigger_bram": block},
        "fmpad": {"SIMD": n["SWG_m"].simd},
        "swg": {"SIMD": n["SWG_m"].simd, "parallel_window": int(n["MVAU_m"].simd > cmid)},
        "mvau_m": {"PE": n["MVAU_m"].pe, "SIMD": n["MVAU_m"].simd},
        "thr_m": {"PE": n["Thr_m"].pe, "depth_trigger_bram": block},
        "mvau_e": {"PE": n["MVAU_e"].pe, "SIMD": n["MVAU_e"].simd},
        "thr_e": {"PE": n["Thr_e"].pe, "depth_trigger_bram": block},
        "thr_s": {"PE": n["Thr_s"].pe, "depth_trigger_bram": block},
        "add": {"PE": n["Add"].pe},
        "thr_out": {"PE": n["Thr_out"].pe, "depth_trigger_bram": block},
    }
    role = lambda name: "dwc" if name.startswith("DWC(") else _ROLE_OF.get(name, name.lower())
    fifos = []
    for name, f in r.fifo_graph["fifos"].items():
        if f["producer"] == "Source" or f["consumer"] == "Sink":
            continue
        fifos.append(dict(
            name=name, producer=role(f["producer"]), consumer=role(f["consumer"]), producer_node=f["producer"],
            consumer_node=f["consumer"], depth=int(f["depth"]), width_bits=int(f["bits"]), max_occupancy=int(f.get("max_occ", 0)),
            is_skip=name == "skip FIFO",
        ))
    v = r.verification
    return dict(
        params=dict(p), folding=folding, fifos=fifos,
        predicted=dict(
            T=p["T"], steady_cyc_px=v.get("steady_cyc_px"), latency_first_out=v.get("latency_first_out"),
            frame_cycles=v.get("frame_cycles"), skip_fifo_words=r.skip_fifo.depth_words, skip_fifo_width_bits=r.skip_fifo.width_bits,
            totals=dict(r.totals),
            nodes={_ROLE_OF[x.name]: dict(
                op=x.op, pe=x.pe, simd=x.simd, cyc_px=x.cyc_px, lut=x.lut, bram18=x.bram18, uram=x.uram, dsp=x.dsp,
                in_width_bits=x.in_width_bits, out_width_bits=x.out_width_bits) for x in r.nodes},
            dwcs=[dict(edge=d.edge, in_width=d.in_width, out_width=d.out_width, lut=d.lut) for d in r.dwcs],
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cin", type=int, required=True)
    ap.add_argument("--v", type=int, required=True)
    ap.add_argument("--z", type=int, required=True)
    ap.add_argument("--T", type=int, required=True, help="target cycles per output pixel")
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--dilation", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--onnx", metavar="PATH", help="verify with the simulator, then write the dataflow graph (needs the onnx package)")
    ap.add_argument("--verify", action="store_true", help="run the simulation check without writing ONNX")
    ap.add_argument("--folding-json", metavar="PATH", help="verify with the simulator, then write the FINN probe folding + FIFO config")
    ap.add_argument("--fifo-depth", type=int, default=2, help="depth in words of ordinary inter-node FIFOs (simulation / ONNX)")
    a = ap.parse_args()
    r = model_bottleneck(a.cin, a.v, a.z, a.T, a.bits, a.height, a.width, a.k, a.dilation, a.stride)
    if a.onnx or a.verify or a.folding_json:
        verify_with_sim(r, fifo_depth=a.fifo_depth)
        if a.onnx:
            export_onnx(r, a.onnx)
        if a.folding_json:
            with open(a.folding_json, "w") as fh:
                json.dump(to_folding_config(r), fh, indent=2)
    print(json.dumps(r.to_dict(), indent=2) if a.json else r.report())
    if a.onnx:
        print(f"wrote {a.onnx}")
    if a.folding_json:
        print(f"wrote {a.folding_json}")


if __name__ == "__main__":
    main()
