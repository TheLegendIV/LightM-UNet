"""Cycle-level token simulation of the ENet INITIAL block (companion of int_bottleneck.py), FINNInitialBlockConcat:

    Source -> Thr_in -> Dup -> [FMPad (pad 1) -> SWG (3x3, stride 2) -> MVAU_c (Cin*9 -> Cout-Cin) -> Thr_c (branch_quant)] -> FIFO main -+
                              [Thr_m (branch_quant) -> MaxPool (2x2, stride 2)] ------------------------------------------> skip FIFO -+-> Concat -> Thr_act

Thr_m is upstream of the pool (landed FINN graph: MoveMaxPoolPastMultiThreshold swaps MaxPool -> Thr into Thr -> MaxPool).

Pixel domains: Source, Thr_in, Dup, FMPad, SWG input side, Thr_m and MaxPool see the H x W input; everything after the stride the (H/2) x (W/2) output.
New node type: SwgStrNode, a strided KxK window generator over a padded image (the stride-2 3x3 of the conv branch). MaxPoolNode comes from
dn_bottleneck_sim, the rest from bottleneck_sim.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bottleneck_sim import (  # noqa: E402
    BLOCKED, BUSY, IDLE, STARVED, UNBOUNDED, FmPadNode, Fifo, MvauNode, Sink, Source, SimResult, StreamNode, run_network,
)
from dn_bottleneck_sim import MaxPoolNode, Swg2Node, swg_cap_px  # noqa: E402


class SwgStrNode:
    """KxK window with stride s over a (hin x win) PADDED image (input pixel ids are LOCAL, as from FmPadNode); windows ho x wo."""

    def __init__(self, name, inp: Fifo, out: Fifo, hin: int, win: int, k: int, s: int, cf: int, out_w: int, cap_px: int, frames: int = 1):
        self.name, self.inp, self.out, self.hin, self.win, self.k, self.s, self.cf, self.out_w = name, inp, out, hin, win, k, s, cf, out_w
        self.ho, self.wo = (hin - k) // s + 1, (win - k) // s + 1
        self.cap_words = cap_px * cf
        self.total_in = hin * win * cf
        self.frames, self.f = frames, 0
        self.recv = self.u = self.e = 0

    def _base_last(self):
        oy, ox = divmod(self.u, self.wo)
        return oy * self.s * self.win + ox * self.s, (oy * self.s + self.k - 1) * self.win + ox * self.s + self.k - 1

    def step(self, t: int) -> int:
        # The strided window never reads the last padded row / column (258 padded, last window ends at column 256), so after the last window
        # was emitted the node still has to swallow the rest of the padded frame before the next frame starts.
        progressed = blocked = False
        done = self.f >= self.frames
        emit_done = self.u >= self.ho * self.wo
        base = self._base_last()[0] if not (done or emit_done) else self.hin * self.win
        if not done and self.recv < self.total_in and self.inp.q and self.recv - base * self.cf < self.cap_words:
            q, _ = self.inp.q.popleft()
            assert q == self.recv // self.cf, f"{self.name}: input pixel {q}, expected {self.recv // self.cf}"
            self.recv += 1
            progressed = True
        if not done and not emit_done:
            base, last = self._base_last()
            if self.recv >= (last + 1) * self.cf:
                if self.out.space():
                    self.out.push((self.f * self.ho * self.wo + self.u, self.e))
                    self.e += 1
                    progressed = True
                    if self.e == self.out_w:
                        self.e, self.u = 0, self.u + 1
                else:
                    blocked = True
        if not done and self.u >= self.ho * self.wo and self.recv >= self.total_in:
            self.u, self.recv, self.f = 0, 0, self.f + 1
        if progressed:
            return BUSY
        if blocked:
            return BLOCKED
        return IDLE if self.f >= self.frames else STARVED


def simulate_int(
    r, inject_interval: float = 0, skip_depth: int | None = None, main_depth: int | None = None, fifo_depth: int = 2,
    fifo_depths: dict | None = None, frames: int = 1, elastic_map: dict | None = None, max_cycles: int | None = None,
) -> SimResult:
    """skip_depth: 'skip FIFO' = end of the MAXPOOL branch, main_depth: 'FIFO main' = end of the conv branch (words; None = r's value,
    UNBOUNDED = measure). inject_interval: cycles between INPUT pixels."""
    p = r.params
    H, W, Ho, Wo = p["height"], p["width"], p["hout"], p["wout"]
    cin, cout = p["cin"], p["cout"]
    ccv = cout - cin
    n_in, n_out = H * W * frames, Ho * Wo * frames
    n = {x.name: x for x in r.nodes}
    order: list = []
    fifos: dict = {}
    if skip_depth is None:
        skip_depth = r.skip_fifo.depth_words
    if main_depth is None:
        main_depth = p.get("main_fifo_words", 2)

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

    thr_in_w = cin // n["Thr_in"].pe
    f_src = fifo("src->Thr_in")
    order.append(Source("Source", f_src, n_in, thr_in_w, inject_interval))
    f_t = fifo("Thr_in->out")
    order.append(StreamNode("Thr_in", [f_src], [f_t], thr_in_w, thr_in_w, n_in))
    dup_w = cin // n["Dup"].pe
    f = link("Thr_in", thr_in_w, "Dup", dup_w, f_t, n_in)
    f_dc, f_dm = fifo("Dup->conv"), fifo("Dup->pool")
    order.append(StreamNode("Dup", [f], [f_dc, f_dm], dup_w, dup_w, n_in))

    # ---- conv branch
    cf = cin // n["SWG"].simd
    sf, nf, thr_c_w = 9 * cin // n["MVAU_c"].simd, ccv // n["MVAU_c"].pe, ccv // n["Thr_c"].pe
    f = link("Dup", dup_w, "FMPad", cf, f_dc, n_in)
    f_o = fifo("FMPad->out")
    order.append(FmPadNode("FMPad", f, f_o, H, W, 1, cf, frames))
    f_o2 = fifo("SWG->out")
    cap_px = 3 * (W + 2) - 6                              # finn_cost_model._finn_swu: buffer_min + (sh-1)*win - kh*kw, win = padded width
    par = n["MVAU_c"].simd > cin                            # parallel_window: the SWG emits one window (cf words) per output pixel, a DWC splits it into sf words
    order.append(SwgStrNode("SWG", f_o, f_o2, H + 2, W + 2, 3, 2, cf, cf if par else sf, cap_px, frames))
    f = link("SWG", cf if par else sf, "MVAU_c", sf, f_o2, n_out)
    f_o = fifo("MVAU_c->out")
    order.append(MvauNode("MVAU_c", f, f_o, sf, nf, n_out))
    f = link("MVAU_c", nf, "Thr_c", thr_c_w, f_o, n_out)
    f_main = fifo("FIFO main", main_depth)
    order.append(StreamNode("Thr_c", [f], [f_main], thr_c_w, thr_c_w, n_out))

    # ---- pool branch: Thr_m (branch quant, full resolution) -> MaxPool
    thr_m_w = cin // n["Thr_m"].pe
    f = link("Dup", dup_w, "Thr_m", thr_m_w, f_dm, n_in)
    f_tm = fifo("Thr_m->out")
    order.append(StreamNode("Thr_m", [f], [f_tm], thr_m_w, thr_m_w, n_in))
    f_skip = fifo("skip FIFO", skip_depth)
    if p.get("pool_impl", "streaming") == "streaming":
        f = link("Thr_m", thr_m_w, "MaxPool", 1, f_tm, n_in)
        order.append(MaxPoolNode("MaxPool", f, f_skip, H, W, frames))
        pool_w = 1
    else:          # InferPool route: depthwise 2x2 stride-2 SWG (SIMD = PE) -> Pool_hls (PE)
        cf_p = cin // n["SWG_p"].simd
        f = link("Thr_m", thr_m_w, "SWG_p", cf_p, f_tm, n_in)
        f_o = fifo("SWG_p->out")
        pw = bool(p.get("pool_pw"))
        order.append(Swg2Node("SWG_p", f, f_o, H, W, cf_p, cf_p if pw else 4 * cf_p, swg_cap_px(W, 2, 2, 2, 2), frames))
        f_o = link("SWG_p", cf_p, "Pool", 4 * cf_p, f_o, n_out) if pw else f_o       # parallel_window: one window per word, a DWC splits it for Pool_hls
        order.append(StreamNode("Pool", [f_o], [f_skip], 4 * cf_p, cf_p, n_out))
        pool_w = cf_p

    # ---- concat (one output pixel per cycle: one word of each branch) -> activation threshold
    thr_a_w = cout // n["Thr_act"].pe
    f_a = link("FIFOmain", thr_c_w, "Concat", 1, f_main, n_out)
    f_b = link("skipFIFO", pool_w, "Concat", 1, f_skip, n_out)
    f_cat = fifo("Concat->out")
    order.append(StreamNode("Concat", [f_a, f_b], [f_cat], 1, 1, n_out))
    f = link("Concat", 1, "Thr_act", thr_a_w, f_cat, n_out)
    f_out = fifo("Thr_act->out")
    order.append(StreamNode("Thr_act", [f], [f_out], thr_a_w, thr_a_w, n_out))
    sink = Sink("Sink", f_out, thr_a_w)
    order.append(sink)

    return run_network(
        r, order, fifos, sink, n_out, inject_interval, skip_depth, fifo_depth, None, fifo_depths, max_cycles,
        px_per_frame=(Ho * Wo if frames > 1 else None), elastic_map=elastic_map,
    )
