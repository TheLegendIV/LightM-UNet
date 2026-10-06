"""Cycle-level token simulation of one ENet bottleneck (companion of bottleneck.py).

It builds the same node chain / folding that bottleneck.model_bottleneck() chose, injects
pixel tokens (pixel id, word index) into a Dup node and steps every node once per clock.
Every edge is a finite FIFO (backpressure), words move at <= 1 per cycle per port.
Per cycle each node reports BUSY / STARVED (input missing) / BLOCKED (output full) / IDLE.
Results: steady-state cyc/px, first-pixel latency, per-node utilisation over time,
max FIFO occupancy (the skip FIFO depth that is actually needed) and deadlock detection.

Node behaviour (all words are one stream word of a node's own width):
  MVAU      pass 0 (SF cycles) consumes SF input words, every pass of SF cycles emits one
            output word; a pixel takes NF*SF steps; a step stalls on missing input / full output.
  FMPad     emits the padded image in raster order; real pixels pass through, padding is free.
  SWG       line buffer: a window is emitted (SF words) once its last input pixel arrived;
            input stalls when the buffer (span + slack pixels) is full.
  stream    Thresholding / Add / Dup / DWC: II = 1, per-pixel word counts in -> out.
Tokens carry the pixel id and every consumer checks it, so a misaligned skip/main join fails loudly.

Run: python3 bottleneck_sim.py --cin 32 --v 4 --z 4 --T 72 --bits 4 --height 32 --width 32 --k 3 --dilation 8
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import deque
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bottleneck import BottleneckResult, model_bottleneck  # noqa: E402

BUSY, STARVED, BLOCKED, IDLE = 0, 1, 2, 3
STATE_NAMES = ("busy", "starved", "blocked", "idle")
UNBOUNDED = 10 ** 9
CAPTURE: dict = {"out": None}     # set "out" to a list: run_network then only builds the network, appends it there and returns None (net_fifo.py)


class Fifo:
    def __init__(self, name: str, depth: int):
        self.name, self.depth, self.q, self.max_occ = name, depth, deque(), 0

    def space(self) -> bool:
        return len(self.q) < self.depth

    def push(self, tok) -> None:
        self.q.append(tok)
        if len(self.q) > self.max_occ:
            self.max_occ = len(self.q)


class Source:
    def __init__(self, name, out: Fifo, n_px: int, words: int, interval: int):
        self.name, self.out, self.n, self.words, self.interval = name, out, n_px, words, interval
        self.p = self.w = 0

    def step(self, t: int) -> int:
        if self.p >= self.n or t < self.p * self.interval:
            return IDLE
        if not self.out.space():
            return BLOCKED
        self.out.push((self.p, self.w))
        self.w += 1
        if self.w == self.words:
            self.w, self.p = 0, self.p + 1
        return BUSY


class Sink:
    def __init__(self, name, inp: Fifo, words: int):
        self.name, self.inp, self.words, self.done = name, inp, words, []

    def step(self, t: int) -> int:
        if not self.inp.q:
            return STARVED
        pid, w = self.inp.q.popleft()
        if w == self.words - 1:
            assert pid == len(self.done), f"{self.name}: pixel {pid} out of order (expected {len(self.done)})"
            self.done.append(t)
        return BUSY


class StreamNode:
    """II = 1 node. Per pixel it consumes in_w words (from every input) and emits out_w words (to every output)."""

    def __init__(self, name, ins: list, outs: list, in_w: int, out_w: int, n_px: int):
        self.name, self.ins, self.outs, self.in_w, self.out_w, self.n = name, ins, outs, in_w, out_w, n_px
        self.I = self.O = 0
        self.total_in, self.total_out = n_px * in_w, n_px * out_w
        self.chunk = max(1, math.ceil(in_w / out_w))

    def _needed(self, O: int) -> int:
        p, j = divmod(O, self.out_w)
        return p * self.in_w + -(-(j + 1) * self.in_w // self.out_w)

    def step(self, t: int) -> int:
        progressed = False
        can_emit_input = self.O < self.total_out and self.I >= self._needed(self.O)
        emit_blocked = False
        if can_emit_input:
            if all(o.space() for o in self.outs):
                p, j = divmod(self.O, self.out_w)
                for o in self.outs:
                    o.push((p, j))
                self.O += 1
                progressed = True
            else:
                emit_blocked = True
        limit = (self._needed(self.O) if self.O < self.total_out else self.total_in) + self.chunk
        if self.I < self.total_in and self.I < limit and all(f.q for f in self.ins):
            expect = self.I // self.in_w
            for f in self.ins:
                pid, _ = f.q.popleft()
                assert pid == expect, f"{self.name}: token for pixel {pid}, expected {expect}"
            self.I += 1
            progressed = True
        if progressed:
            return BUSY
        if emit_blocked:
            return BLOCKED
        if self.O >= self.total_out and self.I >= self.total_in:
            return IDLE
        return STARVED


class MvauNode:
    def __init__(self, name, inp: Fifo, out: Fifo, sf: int, nf: int, n_px: int):
        self.name, self.inp, self.out, self.sf, self.nf, self.n = name, inp, out, sf, nf, n_px
        self.p = self.s = 0

    def step(self, t: int) -> int:
        if self.p >= self.n:
            return IDLE
        if self.s < self.sf and not self.inp.q:
            return STARVED
        end_of_pass = (self.s + 1) % self.sf == 0
        if end_of_pass and not self.out.space():
            return BLOCKED
        if self.s < self.sf:
            pid, _ = self.inp.q.popleft()
            assert pid == self.p, f"{self.name}: token for pixel {pid}, expected {self.p}"
        if end_of_pass:
            self.out.push((self.p, (self.s + 1) // self.sf - 1))
        self.s += 1
        if self.s == self.sf * self.nf:
            self.s, self.p = 0, self.p + 1
        return BUSY


class FmPadNode:
    def __init__(self, name, inp: Fifo, out: Fifo, h: int, w: int, pad: int, cf: int, frames: int = 1):
        self.name, self.inp, self.out, self.h, self.w, self.pad, self.cf = name, inp, out, h, w, pad, cf
        self.hp, self.wp = h + 2 * pad, w + 2 * pad
        self.q = self.wd = 0
        self.frames, self.f = frames, 0

    def step(self, t: int) -> int:
        if self.f >= self.frames:
            return IDLE
        r, c = divmod(self.q, self.wp)
        real = self.pad <= r < self.pad + self.h and self.pad <= c < self.pad + self.w
        if real and not self.inp.q:
            return STARVED
        if not self.out.space():
            return BLOCKED
        if real:
            pid, _ = self.inp.q.popleft()
            expect = self.f * self.h * self.w + (r - self.pad) * self.w + (c - self.pad)
            assert pid == expect, f"{self.name}: input pixel {pid}, expected {expect}"
        self.out.push((self.q, self.wd))
        self.wd += 1
        if self.wd == self.cf:
            self.wd, self.q = 0, self.q + 1
            if self.q == self.hp * self.wp:          # padded frame complete: the next frame starts right away
                self.q, self.f = 0, self.f + 1
        return BUSY


class SwgNode:
    """Sliding-window generator: stride 1, same padding, dilated k_eff window."""

    def __init__(self, name, inp: Fifo, out: Fifo, h: int, w: int, k_eff: int, cf: int, out_w: int, slack_px: int = 1,
                 frames: int = 1):
        self.name, self.inp, self.out, self.h, self.w, self.k_eff, self.cf, self.out_w = name, inp, out, h, w, k_eff, cf, out_w
        self.frames, self.f = frames, 0
        self.pad = (k_eff - 1) // 2
        self.wp = w + 2 * self.pad
        self.hp = h + 2 * self.pad
        self.cap_words = ((k_eff - 1) * self.wp + k_eff + slack_px) * cf
        self.recv = 0          # input words received
        self.u = 0             # current output window
        self.e = 0             # words of window u already emitted
        self.total_in = self.hp * self.wp * cf

    def _base_last(self):
        oy, ox = divmod(self.u, self.w)
        return oy * self.wp + ox, (oy + self.k_eff - 1) * self.wp + ox + self.k_eff - 1

    def step(self, t: int) -> int:
        progressed = False
        blocked = False
        done = self.f >= self.frames
        base = self._base_last()[0] if not done else self.hp * self.wp
        if not done and self.recv < self.total_in and self.inp.q and self.recv - base * self.cf < self.cap_words:
            q, _ = self.inp.q.popleft()
            assert q == self.recv // self.cf, f"{self.name}: input pixel {q}, expected {self.recv // self.cf}"
            self.recv += 1
            progressed = True
        if not done:
            base, last = self._base_last()
            if self.recv >= (last + 1) * self.cf:
                if self.out.space():
                    self.out.push((self.f * self.h * self.w + self.u, self.e))
                    self.e += 1
                    progressed = True
                    if self.e == self.out_w:
                        self.e, self.u = 0, self.u + 1
                        if self.u == self.h * self.w:    # last window of the frame emitted -> next frame
                            self.u, self.recv, self.f = 0, 0, self.f + 1
                else:
                    blocked = True
        if progressed:
            return BUSY
        if blocked:
            return BLOCKED
        return IDLE if self.f >= self.frames else STARVED


# ---------------------------------------------------------------- build + run

def _io(node) -> tuple[list, list]:
    if hasattr(node, "ins"):
        return list(node.ins), list(node.outs)
    if isinstance(node, Source):
        return [], [node.out]
    if isinstance(node, Sink):
        return [node.inp], []
    return [node.inp], [node.out]


def _describe_graph(r: BottleneckResult, order: list, fifos: dict) -> dict:
    """Topology of the simulated network: node order, per-node fifo names, and each FIFO's depth / word width / endpoints."""
    by_name = {x.name: x for x in r.nodes}
    producer, consumer = {}, {}
    io = {}
    for node in order:
        ins, outs = _io(node)
        io[node.name] = ([f.name for f in ins], [f.name for f in outs])
        for f in outs:
            producer[f.name] = node.name
        for f in ins:
            consumer[f.name] = node.name

    def out_bits(prod: str) -> int:
        if prod in by_name:
            return by_name[prod].out_width_bits
        if prod.startswith("DWC("):
            return by_name[prod[4:-1].split("->")[1]].in_width_bits
        if prod == "Source":
            return by_name["Dup"].in_width_bits if "Dup" in by_name else r.nodes[0].in_width_bits
        raise KeyError(prod)

    return dict(
        nodes=[n.name for n in order], io=io,
        fifos={
            f.name: dict(depth=f.depth, bits=out_bits(producer[f.name]), producer=producer[f.name], consumer=consumer[f.name])
            for f in fifos.values()
        },
    )


@dataclass
class SimResult:
    deadlock: bool
    cycles: int
    out_times: list
    first_inject: int
    latency_first_out: int
    steady_cyc_px: float
    node_names: list
    states: dict           # node name -> bytearray of per-cycle states
    steady_window: tuple   # (t_a, t_b) cycle range used for steady statistics
    fifo_max: dict
    params: dict = field(default_factory=dict)
    graph: dict = field(default_factory=dict)  # nodes (topological), io, fifos (depth/bits/producer/consumer)
    frame_periods: list = field(default_factory=list)   # cycles between consecutive frame completions (frames > 1)
    full_fifos: list = field(default_factory=list)      # names of the FIFOs at capacity when the run ended (the blocked chain of a deadlock)

    def fractions(self, name: str, window: tuple | None = None) -> dict:
        a, b = window if window else (0, self.cycles)
        seg = self.states[name][a:b]
        n = max(1, len(seg))
        return {STATE_NAMES[i]: seg.count(i) / n for i in range(4)}


def _words(r: BottleneckResult) -> dict:
    p = r.params
    n = {x.name: x for x in r.nodes}
    k, cin, cmid, cout = p["k"], p["cin"], p["cmid"], p["cout"]
    w = dict(
        cin=cin, cmid=cmid, cout=cout, k=k,
        dup=cin // n["Dup"].pe,
        sf_r=cin // n["MVAU_r"].simd, nf_r=cmid // n["MVAU_r"].pe, thr_r=cmid // n["Thr_r"].pe,
        simd_swu=n["SWG_m"].simd, cf=cmid // n["SWG_m"].simd,
        sf_m=(k * k * cmid) // n["MVAU_m"].simd, nf_m=cmid // n["MVAU_m"].pe, thr_m=cmid // n["Thr_m"].pe,
        sf_e=cmid // n["MVAU_e"].simd, nf_e=cout // n["MVAU_e"].pe, thr_e=cout // n["Thr_e"].pe,
        thr_s=cout // n["Thr_s"].pe, add=cout // n["Add"].pe, thr_out=cout // n["Thr_out"].pe,
    )
    return w


def simulate(
    r: BottleneckResult, inject_interval: int = 0, skip_depth: int | None = None, fifo_depth: int = 2,
    swg_slack_px: int = 1, max_cycles: int | None = None, elastic_depth: int | None = None,
    fifo_depths: dict | None = None, frames: int = 1,
) -> SimResult:
    """frames: number of consecutive frames fed back to back (steady_cyc_px is then the LAST frame's period per pixel).
    inject_interval: cycles between input pixels (0 = as fast as the chain accepts, T = upstream block at rate T).
    skip_depth: skip FIFO depth in words (None = the analytic depth of r, UNBOUNDED = measure the need).
    elastic_depth: depth in words of the FIFO feeding FMPad (None = fifo_depth). It must hold the pad+1 real pixels the
    SWG needs at each output row start; see bottleneck.verify_with_sim. fifo_depths: per-FIFO overrides (name -> words),
    applied last."""
    p = r.params
    H, W, pad = p["height"], p["width"], p["pad"]
    N = H * W * frames
    k_eff = (p["k"] - 1) * p["dilation"] + 1
    w = _words(r)
    if skip_depth is None:
        skip_depth = r.skip_fifo.depth_words

    fifos: dict[str, Fifo] = {}
    order: list = []
    conv_count = [0]

    def fifo(name, depth=None):
        f = Fifo(name, depth or fifo_depth)
        fifos[name] = f
        return f

    def link(prev_name, prev_out_w, next_name, next_in_w, src_fifo: Fifo, n_px: int, depth=None) -> Fifo:
        """Fifo from a producer to a consumer; inserts a DWC node (stream converter) on a word-count mismatch."""
        if prev_out_w == next_in_w:
            return src_fifo
        conv_count[0] += 1
        nxt = fifo(f"{prev_name}->DWC->{next_name}")
        order.append(StreamNode(f"DWC({prev_name}->{next_name})", [src_fifo], [nxt], prev_out_w, next_in_w, n_px))
        return nxt

    # source and dup
    f_src = fifo("src->Dup")
    order.append(Source("Source", f_src, N, w["dup"], inject_interval))
    f_dup_main, f_dup_skip = fifo("Dup->main"), fifo("Dup->skip")
    order.append(StreamNode("Dup", [f_src], [f_dup_main, f_dup_skip], w["dup"], w["dup"], N))

    # main branch
    f = link("Dup", w["dup"], "MVAU_r", w["sf_r"], f_dup_main, N)
    f_o = fifo("MVAU_r->out")
    order.append(MvauNode("MVAU_r", f, f_o, w["sf_r"], w["nf_r"], N))
    f = f_o
    f_o = fifo("Thr_r->out")
    order.append(StreamNode("Thr_r", [link("MVAU_r", w["nf_r"], "Thr_r", w["thr_r"], f, N)], [f_o], w["thr_r"], w["thr_r"], N))
    f = link("Thr_r", w["thr_r"], "FMPad", w["cf"], f_o, N)
    f_o = fifo("FMPad->out")
    order.append(FmPadNode("FMPad", f, f_o, H, W, pad, w["cf"], frames))
    f_o2 = fifo("SWG_m->out")
    order.append(SwgNode("SWG_m", f_o, f_o2, H, W, k_eff, w["cf"], w["sf_m"], swg_slack_px, frames))
    f = link("SWG_m", w["sf_m"], "MVAU_m", w["sf_m"], f_o2, N)
    f_o = fifo("MVAU_m->out")
    order.append(MvauNode("MVAU_m", f, f_o, w["sf_m"], w["nf_m"], N))
    f = link("MVAU_m", w["nf_m"], "Thr_m", w["thr_m"], f_o, N)
    f_o = fifo("Thr_m->out")
    order.append(StreamNode("Thr_m", [f], [f_o], w["thr_m"], w["thr_m"], N))
    f = link("Thr_m", w["thr_m"], "MVAU_e", w["sf_e"], f_o, N)
    f_o = fifo("MVAU_e->out")
    order.append(MvauNode("MVAU_e", f, f_o, w["sf_e"], w["nf_e"], N))
    f = link("MVAU_e", w["nf_e"], "Thr_e", w["thr_e"], f_o, N)
    f_o = fifo("Thr_e->out")
    order.append(StreamNode("Thr_e", [f], [f_o], w["thr_e"], w["thr_e"], N))
    f_main_add = link("Thr_e", w["thr_e"], "Add", w["add"], f_o, N)

    # skip branch
    f = link("Dup", w["dup"], "Thr_s", w["thr_s"], f_dup_skip, N)
    f_skip = fifo("skip FIFO", skip_depth)
    order.append(StreamNode("Thr_s", [f], [f_skip], w["thr_s"], w["thr_s"], N))
    f_skip_add = link("skipFIFO", w["thr_s"], "Add", w["add"], f_skip, N)   # widening DWC sits AFTER the FIFO

    # join
    f_add = fifo("Add->out")
    order.append(StreamNode("Add", [f_main_add, f_skip_add], [f_add], w["add"], w["add"], N))
    f = link("Add", w["add"], "Thr_out", w["thr_out"], f_add, N)
    f_out = fifo("Thr_out->out")
    order.append(StreamNode("Thr_out", [f], [f_out], w["thr_out"], w["thr_out"], N))
    sink = Sink("Sink", f_out, w["thr_out"])
    order.append(sink)

    return run_network(
        r, order, fifos, sink, N, inject_interval, skip_depth, fifo_depth, elastic_depth, fifo_depths, max_cycles,
        px_per_frame=(H * W if frames > 1 else None),
    )


def run_network(
    r, order: list, fifos: dict, sink, N: int, inject_interval: int, skip_depth: int, fifo_depth: int,
    elastic_depth=None, fifo_depths=None, max_cycles=None, elastic_consumers=("FMPad",), px_per_frame=None,
    elastic_map=None,
) -> SimResult:
    """Step a built network (nodes in topological order) until the sink has N output pixels. Shared by every block type;
    elastic_depth is applied to the FIFOs whose consumer node is named in elastic_consumers."""
    graph = _describe_graph(r, order, fifos)
    if elastic_depth is not None:
        for f in fifos.values():
            if graph["fifos"][f.name]["consumer"] in elastic_consumers:
                f.depth = elastic_depth
                graph["fifos"][f.name]["depth"] = elastic_depth
    for f in fifos.values():                       # elastic_map: consumer node name -> depth of the FIFO(s) feeding it
        c = graph["fifos"][f.name]["consumer"]
        if elastic_map and c in elastic_map:
            f.depth = graph["fifos"][f.name]["depth"] = elastic_map[c]
    for name, d in (fifo_depths or {}).items():
        fifos[name].depth = d
        graph["fifos"][name]["depth"] = d
    if CAPTURE["out"] is not None:                 # net_fifo: hand the built (not yet run) network to the chain driver
        CAPTURE["out"].append(dict(order=order, fifos=fifos, sink=sink, N=N, graph=graph, px_per_frame=px_per_frame))
        return None

    names = [n.name for n in order]
    limit = max_cycles or int(6 * (r.frame_cycles + N * max(inject_interval, 1)) + 20000)
    states = {n: bytearray(limit) for n in names}
    t = 0
    last_progress = 0
    deadlock = False
    rev = list(reversed(order))
    while len(sink.done) < N:
        progressed = False
        for node in rev:
            s = node.step(t)
            states[node.name][t] = s
            if s == BUSY:
                progressed = True
        if progressed:
            last_progress = t
        elif t - last_progress > 5000 + 2 * inject_interval:
            deadlock = True
            break
        t += 1
        if t >= limit:
            deadlock = True
            break
    cycles = t
    states = {n: s[:cycles] for n, s in states.items()}
    done = sink.done
    if done:
        a, b = done[int(0.2 * len(done))], done[max(int(0.2 * len(done)) + 1, int(0.8 * len(done)) - 1)]
        steady = (b - a) / max(1, (int(0.8 * len(done)) - 1) - int(0.2 * len(done)))
    else:
        a, b, steady = 0, cycles, float("inf")
    periods = []
    if px_per_frame and len(done) >= 2 * px_per_frame:
        ends = [done[(k + 1) * px_per_frame - 1] for k in range(len(done) // px_per_frame)]
        periods = [ends[k] - ends[k - 1] for k in range(1, len(ends))]
        steady = periods[-1] / px_per_frame
    return SimResult(
        deadlock=deadlock, cycles=cycles, out_times=done, first_inject=0,
        latency_first_out=(done[0] if done else -1), steady_cyc_px=steady, node_names=names, states=states,
        steady_window=(a, b), fifo_max={n: f.max_occ for n, f in fifos.items()}, params=dict(
            inject_interval=inject_interval, skip_depth=skip_depth, fifo_depth=fifo_depth),
        graph=graph, frame_periods=periods, full_fifos=[n for n, f in fifos.items() if len(f.q) >= f.depth],
    )


# ---------------------------------------------------------------- reporting

_SHADES = " .:-=+*#%@"


def timeline(res: SimResult, name: str, cols: int = 60) -> str:
    seg = res.states[name]
    step = max(1, len(seg) // cols)
    out = []
    for i in range(0, len(seg), step):
        chunk = seg[i:i + step]
        out.append(_SHADES[min(9, int(chunk.count(BUSY) / len(chunk) * 9.999))])
    return "".join(out[:cols])


def report(res: SimResult, T: int, title: str = "") -> str:
    lines = [title] if title else []
    if res.deadlock:
        lines.append(f"DEADLOCK / stall after {res.cycles} cycles, {len(res.out_times)} pixels out")
    lines.append(
        f"pixels out {len(res.out_times)}  frame {res.cycles} cycles  first-out latency {res.latency_first_out}  "
        f"steady {res.steady_cyc_px:.2f} cyc/px (target {T})"
    )
    lines.append(f"{'node':26s} {'busy':>6s} {'starv':>6s} {'block':>6s} {'idle':>6s}   busy over time (steady window busy/starved/blocked)")
    for n in res.node_names:
        if n in ("Source", "Sink"):
            continue
        fr = res.fractions(n, res.steady_window)
        lines.append(
            f"{n:26s} {fr['busy']:6.2f} {fr['starved']:6.2f} {fr['blocked']:6.2f} {fr['idle']:6.2f}   |{timeline(res, n)}|"
        )
    return "\n".join(lines)


def analyze(r: BottleneckResult, fifo_depth: int = 2) -> dict:
    """Saturated run (true max throughput + bottleneck), paced run at T (latency, skip FIFO need),
    and a verification run with the analytic skip FIFO depth."""
    T = r.params["T"]
    sat = simulate(r, inject_interval=0, skip_depth=UNBOUNDED, fifo_depth=fifo_depth)
    paced = simulate(r, inject_interval=T, skip_depth=UNBOUNDED, fifo_depth=fifo_depth)
    need = paced.fifo_max.get("skip FIFO", 0)
    check = simulate(r, inject_interval=T, skip_depth=r.skip_fifo.depth_words, fifo_depth=fifo_depth)
    return dict(saturated=sat, paced=paced, check=check, skip_needed=need)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("cin", "v", "z", "T", "bits", "height", "width"):
        ap.add_argument(f"--{a}", type=int, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--dilation", type=int, default=1)
    ap.add_argument("--fifo-depth", type=int, default=2, help="depth in words of every inter-node FIFO")
    a = ap.parse_args()
    r = model_bottleneck(a.cin, a.v, a.z, a.T, a.bits, a.height, a.width, a.k, a.dilation, 1)
    res = analyze(r, a.fifo_depth)
    T = a.T
    print(report(res["saturated"], T, "== saturated input (max throughput, bottleneck search) =="))
    print()
    print(report(res["paced"], T, f"== input paced at T={T} cyc/px (upstream block rate), unbounded skip FIFO =="))
    print()
    print(f"skip FIFO: simulation needs {res['skip_needed']} words (analytic model {r.skip_fifo.depth_words})")
    c = res["check"]
    print(f"with analytic depth: {'DEADLOCK' if c.deadlock else 'ok'}, steady {c.steady_cyc_px:.2f} cyc/px, frame {c.cycles}")
    print(f"latency first-out: sim {res['paced'].latency_first_out}  analytic {r.latency_first_out_cycles}")
    print(f"max occupancy (paced): " + ", ".join(f"{n}={v}" for n, v in res['paced'].fifo_max.items() if v > a.fifo_depth))


if __name__ == "__main__":
    main()
