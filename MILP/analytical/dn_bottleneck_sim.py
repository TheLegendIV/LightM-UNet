"""Cycle-level token simulation of the ENet DOWNSAMPLING bottleneck (companion of dn_bottleneck.py).

Reuses the node primitives, FIFOs, state recording and run loop of bottleneck_sim.py and adds the three node types the
downsampling block needs:
  MaxPoolNode   StreamingMaxPool: one input pixel (all channels) per cycle, one extra cycle per completed 2x2 window.
  Swg2Node      sliding window of the strided 2x2 reduce conv (kernel 2, stride 2, no padding): a window is ready when its
                bottom-right pixel arrived; the real stride-2 buffer holds (2W-2) pixels (finn_cost_model._finn_swu).
  ChanPadNode   FMPadding used as a CHANNEL pad: per pixel it passes G words through and then emits P zero words
                (stream regrouped as H*W rows x C/s columns x s channels, see analytical.md).

Pixel domains: Source/Dup/SWG_r/MaxPool work on the H x W input pixels, everything after the stride on (H/2) x (W/2).
Sink pixel completions are output pixels, so steady_cyc_px is cycles per OUTPUT pixel (compare with T_out = F / (H/2*W/2)).
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bottleneck_sim import (  # noqa: E402
    BLOCKED, BUSY, IDLE, STARVED, UNBOUNDED, FmPadNode, Fifo, MvauNode, Sink, Source, SimResult, StreamNode, SwgNode,
    run_network,
)


class MaxPoolNode:
    """Input: one word (all channels) per input pixel, raster order. Output: one word per 2x2 window."""

    def __init__(self, name, inp: Fifo, out: Fifo, h: int, w: int, frames: int = 1):
        self.name, self.inp, self.out, self.h, self.w, self.frames = name, inp, out, h, w, frames
        self.i = 0
        self.pending = None

    def step(self, t: int) -> int:
        if self.pending is not None:
            if not self.out.space():
                return BLOCKED
            self.out.push((self.pending, 0))
            self.pending = None
            return BUSY
        if self.i >= self.h * self.w * self.frames:
            return IDLE
        if not self.inp.q:
            return STARVED
        pid, _ = self.inp.q.popleft()
        assert pid == self.i, f"{self.name}: input pixel {pid}, expected {self.i}"
        f, loc = divmod(self.i, self.h * self.w)
        r, c = divmod(loc, self.w)
        self.i += 1
        if r % 2 == 1 and c % 2 == 1:
            self.pending = f * (self.h // 2) * (self.w // 2) + (r // 2) * (self.w // 2) + c // 2
        return BUSY


class Swg2Node:
    """Sliding window for kernel 2, stride 2, no padding. Output pixel (oy, ox) needs input pixels 2oy..2oy+1 x 2ox..2ox+1."""

    def __init__(self, name, inp: Fifo, out: Fifo, h: int, w: int, cf: int, out_w: int, cap_px: int, frames: int = 1):
        self.name, self.inp, self.out, self.h, self.w, self.cf, self.out_w = name, inp, out, h, w, cf, out_w
        self.frames, self.f = frames, 0
        self.wo, self.ho = w // 2, h // 2
        self.cap_words = cap_px * cf
        self.recv = 0
        self.u = 0
        self.e = 0
        self.total_in = h * w * cf

    def _base_last(self):
        oy, ox = divmod(self.u, self.wo)
        return 2 * oy * self.w + 2 * ox, (2 * oy + 1) * self.w + 2 * ox + 1

    def step(self, t: int) -> int:
        progressed = blocked = False
        done = self.f >= self.frames
        base = self._base_last()[0] if not done else self.h * self.w
        if not done and self.recv < self.total_in and self.inp.q and self.recv - base * self.cf < self.cap_words:
            q, _ = self.inp.q.popleft()
            assert q == self.f * self.h * self.w + self.recv // self.cf, f"{self.name}: input pixel {q}, expected {self.recv // self.cf}"
            self.recv += 1
            progressed = True
        if not done:
            base, last = self._base_last()
            if self.recv >= (last + 1) * self.cf:
                if self.out.space():
                    self.out.push((self.f * self.ho * self.wo + self.u, self.e))
                    self.e += 1
                    progressed = True
                    if self.e == self.out_w:
                        self.e, self.u = 0, self.u + 1
                        if self.u == self.ho * self.wo:
                            self.u, self.recv, self.f = 0, 0, self.f + 1
                else:
                    blocked = True
        if progressed:
            return BUSY
        if blocked:
            return BLOCKED
        return IDLE if self.f >= self.frames else STARVED


class ChanPadNode:
    """FMPadding as channel pad: per pixel, g words pass through then p zero words are generated (no input needed)."""

    def __init__(self, name, inp: Fifo, out: Fifo, n_px: int, g: int, p: int):
        self.name, self.inp, self.out, self.n, self.g, self.p = name, inp, out, n_px, g, p
        self.px = 0
        self.j = 0

    def step(self, t: int) -> int:
        if self.px >= self.n:
            return IDLE
        passthrough = self.j < self.g
        if passthrough and not self.inp.q:
            return STARVED
        if not self.out.space():
            return BLOCKED
        if passthrough:
            pid, _ = self.inp.q.popleft()
            assert pid == self.px, f"{self.name}: token for pixel {pid}, expected {self.px}"
        self.out.push((self.px, self.j))
        self.j += 1
        if self.j == self.g + self.p:
            self.j, self.px = 0, self.px + 1
        return BUSY


def _io_names(r):
    return {x.name: x for x in r.nodes}


def swg_cap_px(w: int, kh: int, sh: int, kw: int, sw: int) -> int:
    """finn_cost_model._finn_swu buffer depth (non-parallel window), in pixels (cf factored out)."""
    return ((kh - 1) * w + (kw - 1) + 1) + max(0, (sw - 1) - kh * kw) + max(0, (sh - 1) * w - kh * kw)


def simulate_dn(
    r, inject_interval: float = 0, skip_depth: int | None = None, fifo_depth: int = 2, fifo_depths: dict | None = None,
    elastic_depth: int | None = None, elastic_consumers=("FMPad", "SWG_r"), max_cycles: int | None = None,
    swg_slack_px: int = 1, frames: int = 1, elastic_map: dict | None = None,
) -> SimResult:
    """inject_interval: cycles between INPUT pixels (0 = saturated, r.params['T_in'] = upstream block at the budget)."""
    p = r.params
    H, W, Ho, Wo = p["height"], p["width"], p["height"] // 2, p["width"] // 2
    cin, cmid, cout = p["cin"], p["cmid"], p["cout"]
    n_in, n_out = H * W * frames, Ho * Wo * frames
    n = _io_names(r)
    order: list = []
    fifos: dict = {}

    def fifo(name, depth=None):
        f = Fifo(name, depth or fifo_depth)
        fifos[name] = f
        return f

    def link(prev_name, prev_out_w, next_name, next_in_w, src, n_px):
        if prev_out_w == next_in_w:
            return src
        nxt = fifo(f"{prev_name}->DWC->{next_name}")
        order.append(StreamNode(f"DWC({prev_name}->{next_name})", [src], [nxt], prev_out_w, next_in_w, n_px))
        return nxt

    # word counts per pixel
    dup_w = cin // n["Dup"].pe
    simd_swu_r = n["SWG_r"].simd
    cf_r = cin // simd_swu_r
    sf_r = 4 * cin // n["MVAU_r"].simd
    nf_r = cmid // n["MVAU_r"].pe
    thr_r_w = cmid // n["Thr_r"].pe
    cf_m = cmid // n["SWG_m"].simd
    sf_m = 9 * cmid // n["MVAU_m"].simd
    nf_m = cmid // n["MVAU_m"].pe
    thr_m_w = cmid // n["Thr_m"].pe
    sf_e = cmid // n["MVAU_e"].simd
    nf_e = cout // n["MVAU_e"].pe
    thr_e_w = cout // n["Thr_e"].pe
    add_w = cout // n["Add"].pe
    thr_out_w = cout // n["Thr_out"].pe
    s_pad, skip_order, skip_pad = p.get("pad_group"), p["skip_order"], p["skip_pad"]
    if skip_depth is None:
        skip_depth = r.skip_fifo.depth_words

    f_src = fifo("src->Dup")
    order.append(Source("Source", f_src, n_in, dup_w, inject_interval))
    f_dm, f_ds = fifo("Dup->main"), fifo("Dup->skip")
    order.append(StreamNode("Dup", [f_src], [f_dm, f_ds], dup_w, dup_w, n_in))

    # ---- main branch
    f = link("Dup", dup_w, "SWG_r", cf_r, f_dm, n_in)
    f_o = fifo("SWG_r->out")
    cap = swg_cap_px(W, 2, 2, 2, 2)
    order.append(Swg2Node("SWG_r", f, f_o, H, W, cf_r, sf_r, cap, frames))
    f = link("SWG_r", sf_r, "MVAU_r", sf_r, f_o, n_out)
    f_o = fifo("MVAU_r->out")
    order.append(MvauNode("MVAU_r", f, f_o, sf_r, nf_r, n_out))
    f = link("MVAU_r", nf_r, "Thr_r", thr_r_w, f_o, n_out)
    f_o = fifo("Thr_r->out")
    order.append(StreamNode("Thr_r", [f], [f_o], thr_r_w, thr_r_w, n_out))
    f = link("Thr_r", thr_r_w, "FMPad", cf_m, f_o, n_out)
    f_o = fifo("FMPad->out")
    order.append(FmPadNode("FMPad", f, f_o, Ho, Wo, 1, cf_m, frames))
    f_o2 = fifo("SWG_m->out")
    order.append(SwgNode("SWG_m", f_o, f_o2, Ho, Wo, 3, cf_m, sf_m, swg_slack_px, frames))
    f = link("SWG_m", sf_m, "MVAU_m", sf_m, f_o2, n_out)
    f_o = fifo("MVAU_m->out")
    order.append(MvauNode("MVAU_m", f, f_o, sf_m, nf_m, n_out))
    f = link("MVAU_m", nf_m, "Thr_m", thr_m_w, f_o, n_out)
    f_o = fifo("Thr_m->out")
    order.append(StreamNode("Thr_m", [f], [f_o], thr_m_w, thr_m_w, n_out))
    f = link("Thr_m", thr_m_w, "MVAU_e", sf_e, f_o, n_out)
    f_o = fifo("MVAU_e->out")
    order.append(MvauNode("MVAU_e", f, f_o, sf_e, nf_e, n_out))
    f = link("MVAU_e", nf_e, "Thr_e", thr_e_w, f_o, n_out)
    f_o = fifo("Thr_e->out")
    order.append(StreamNode("Thr_e", [f], [f_o], thr_e_w, thr_e_w, n_out))
    f_main_add = link("Thr_e", thr_e_w, "Add", add_w, f_o, n_out)

    # ---- skip branch: maxpool, then (channel pad | identity MVAU) and the skip requantizing threshold
    if p.get("pool_impl", "streaming") == "streaming":
        f = link("Dup", dup_w, "MaxPool", 1, f_ds, n_in)
        f_mp = fifo("MaxPool->out")
        order.append(MaxPoolNode("MaxPool", f, f_mp, H, W, frames))
        pool_last, pool_w = "MaxPool", 1
    else:          # InferPool route: depthwise 2x2 stride-2 window generator (SIMD = PE) -> Pool_hls (PE), 4*C/PE words in, C/PE words out per pixel
        cf_p = cin // n["SWG_p"].simd
        f = link("Dup", dup_w, "SWG_p", cf_p, f_ds, n_in)
        f_o = fifo("SWG_p->out")
        order.append(Swg2Node("SWG_p", f, f_o, H, W, cf_p, 4 * cf_p, swg_cap_px(W, 2, 2, 2, 2), frames))
        f_mp = fifo("Pool->out")
        order.append(StreamNode("Pool", [f_o], [f_mp], 4 * cf_p, cf_p, n_out))
        pool_last, pool_w = "Pool", cf_p
    thr_s_w = (cin if (skip_pad == "fmpad" and skip_order == "thr_pad") else cout) // n["Thr_s"].pe
    f_skip = fifo("skip FIFO", skip_depth)
    if skip_pad == "mvau":
        sf_s = cin // n["MVAU_s"].simd
        nf_s = cout // n["MVAU_s"].pe
        f = link(pool_last, pool_w, "MVAU_s", sf_s, f_mp, n_out)
        f_o = fifo("MVAU_s->out")
        order.append(MvauNode("MVAU_s", f, f_o, sf_s, nf_s, n_out))
        f = link("MVAU_s", nf_s, "Thr_s", thr_s_w, f_o, n_out)
        order.append(StreamNode("Thr_s", [f], [f_skip], thr_s_w, thr_s_w, n_out))
        f_skip_add = link("skipFIFO", thr_s_w, "Add", add_w, f_skip, n_out)
    else:
        g, pw = cin // s_pad, (cout - cin) // s_pad
        if skip_order == "thr_pad":
            f = link(pool_last, pool_w, "Thr_s", thr_s_w, f_mp, n_out)
            order.append(StreamNode("Thr_s", [f], [f_skip], thr_s_w, thr_s_w, n_out))
            f = link("skipFIFO", thr_s_w, "FMPad_c", g, f_skip, n_out)
            f_pad = fifo("FMPad_c->out")
            order.append(ChanPadNode("FMPad_c", f, f_pad, n_out, g, pw))
            f_skip_add = link("FMPad_c", g + pw, "Add", add_w, f_pad, n_out)
        else:  # pad_thr
            f = link(pool_last, pool_w, "FMPad_c", g, f_mp, n_out)
            f_pad = fifo("FMPad_c->out")
            order.append(ChanPadNode("FMPad_c", f, f_pad, n_out, g, pw))
            f = link("FMPad_c", g + pw, "Thr_s", thr_s_w, f_pad, n_out)
            order.append(StreamNode("Thr_s", [f], [f_skip], thr_s_w, thr_s_w, n_out))
            f_skip_add = link("skipFIFO", thr_s_w, "Add", add_w, f_skip, n_out)

    # ---- join
    f_add = fifo("Add->out")
    order.append(StreamNode("Add", [f_main_add, f_skip_add], [f_add], add_w, add_w, n_out))
    f = link("Add", add_w, "Thr_out", thr_out_w, f_add, n_out)
    f_out = fifo("Thr_out->out")
    order.append(StreamNode("Thr_out", [f], [f_out], thr_out_w, thr_out_w, n_out))
    sink = Sink("Sink", f_out, thr_out_w)
    order.append(sink)

    return run_network(
        r, order, fifos, sink, n_out, inject_interval, skip_depth, fifo_depth, elastic_depth, fifo_depths, max_cycles,
        elastic_consumers=elastic_consumers, px_per_frame=(Ho * Wo if frames > 1 else None), elastic_map=elastic_map,
    )
