"""Cycle-level token simulation of the ENet UPSAMPLING bottleneck, nearest-neighbour decoder (companion of up_bottleneck.py).

Block (FINNUpsamplingBottleneck, decoder_type nearest_conv_upsample; input H x W, output 2H x 2W):
    main:  Dup -> MVAU_p (1x1) -> Thr_p -> UpNN (x2 nearest) -> [FMPad_k -> SWG_k -> MVAU_k (3x3) -> Thr_k] -> FIFO main -+
    ext:   Dup -> MVAU_r (1x1) -> Thr_r -> FMPadPix (zero insertion) -> SWG_u (2x2) -> MVAU_u -> Thr_u -> MVAU_e -> Thr_e -> skip FIFO -+-> Add -> Thr_out
(the [..] conv part is absent for decoder_type nearest_upsample, where UpNN feeds Thr_s -> FIFO main).

New node types (the rest comes from bottleneck_sim / dn_bottleneck_sim):
  UpNNNode        UpsampleNearestNeighbour_hls = finn-hlslib 16e5847 (FINN v0.10.1-10-g39f0c9a6b, the build container) upsample.hpp, line by line:
                  one loop iteration per cycle over (y, x) in 2H x 2W, blocking in.read() / out.write(); a pixel is read only on even output rows y while
                  x < W (RowBuf[x] = in.read()), and output pixel x is RowBuf[x // 2]. So on an even row the first W outputs come out ONE PER INPUT
                  PIXEL, paced by the input (output x leaves when input x has arrived, not x / 2), the last W outputs follow with no read, and the odd
                  row replays the buffer (2W cycles, no read). 4 cycles per input pixel = Hout*Wout when never starved or blocked. A whole iteration
                  stalls when the read starves or the write is blocked. The node is NOT the 2-outputs-per-input emitter of earlier versions of this file:
                  that one made the main branch lead the ext branch, so the skip FIFO was sized for a lead that does not exist (the ext branch leads
                  by up to ~W pixels per row) and the block deadlocked as sized (see up_bottleneck.verify_with_sim, S12_dense_256_u4_analytical_v2).
                  The newer hlslib upsample_nn (8d979e2b, FINN v1.0.0-alpha: frame-sized buffer, non-blocking read) is a different kernel.
  FmPadPixelNode  FMPadding_Pixel + border of a transposed conv (K = S = 2): the (2H+1) x (2W+1) image has a real pixel at every
                  (odd, odd) position; all other positions are zero pixels emitted at 1 word per cycle.
  SwgGenNode      stride-1 window generator for a KxK window without padding over a (hin x win) image (the 2x2 window of the lowered
                  transposed conv); like SwgNode it serves one frame at a time.
Sink pixel completions are OUTPUT pixels (2H x 2W): steady_cyc_px is cycles per output pixel.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bottleneck_sim import (  # noqa: E402
    BLOCKED, BUSY, IDLE, STARVED, UNBOUNDED, FmPadNode, Fifo, MvauNode, Sink, Source, SimResult, StreamNode, SwgNode, run_network,
)


class UpNNNode:
    def __init__(self, name, inp: Fifo, out: Fifo, h: int, w: int, frames: int = 1):
        self.name, self.inp, self.out, self.h, self.w, self.frames = name, inp, out, h, w, frames
        self.f = self.y = self.x = 0                           # frame, output row (0 .. 2H-1), output column (0 .. 2W-1)

    def step(self, t: int) -> int:
        if self.f >= self.frames:
            return IDLE
        reads = self.y % 2 == 0 and self.x < self.w            # hlslib: read_row (even y) && x < IFMDim
        if reads and not self.inp.q:
            return STARVED
        if not self.out.space():
            return BLOCKED
        if reads:
            pid, _ = self.inp.q.popleft()
            expect = (self.f * self.h + self.y // 2) * self.w + self.x
            assert pid == expect, f"{self.name}: input pixel {pid}, expected {expect}"
        self.out.push(((self.f * 2 * self.h + self.y) * 2 * self.w + self.x, 0))
        self.x += 1
        if self.x == 2 * self.w:
            self.x = 0
            self.y += 1
            if self.y == 2 * self.h:
                self.y, self.f = 0, self.f + 1
        return BUSY


class FmPadPixelNode:
    """FMPadding_Pixel: stride-2 zero insertion. edge_pad=True (legacy): ONE node emitting the (2h+1) x (2w+1) grid with the border, real pixels at odd positions, LOCAL ids.
    edge_pad=False (the real FINN graph, partitions 5-7 of the S12-256 build): the (2h-1) x (2w-1) grid, real pixels at even positions, GLOBAL ids; a FmPadNode(pad=1) behind it
    adds the border (FMPadding_rtl)."""

    def __init__(self, name, inp: Fifo, out: Fifo, h: int, w: int, cf: int, frames: int = 1, edge_pad: bool = True):
        self.name, self.inp, self.out, self.h, self.w, self.cf, self.frames, self.edge_pad = name, inp, out, h, w, cf, frames, edge_pad
        self.hp, self.wp = (2 * h + 1, 2 * w + 1) if edge_pad else (2 * h - 1, 2 * w - 1)
        self.q = self.wd = self.f = 0

    def step(self, t: int) -> int:
        if self.f >= self.frames:
            return IDLE
        r, c = divmod(self.q, self.wp)
        off = 1 if self.edge_pad else 0
        real = r % 2 == off and c % 2 == off
        if real and not self.inp.q:
            return STARVED
        if not self.out.space():
            return BLOCKED
        if real:
            pid, _ = self.inp.q.popleft()
            expect = self.f * self.h * self.w + ((r - off) // 2) * self.w + (c - off) // 2
            assert pid == expect, f"{self.name}: input pixel {pid}, expected {expect}"
        self.out.push((self.q if self.edge_pad else self.f * self.hp * self.wp + self.q, self.wd))
        self.wd += 1
        if self.wd == self.cf:
            self.wd, self.q = 0, self.q + 1
            if self.q == self.hp * self.wp:
                self.q, self.f = 0, self.f + 1
        return BUSY


class SwgGenNode:
    """KxK window, stride 1, no padding over a (hin x win) image; input pixels carry LOCAL ids (as from FmPadPixelNode)."""

    def __init__(self, name, inp: Fifo, out: Fifo, hin: int, win: int, k: int, cf: int, out_w: int, slack_px: int = 1, frames: int = 1):
        self.name, self.inp, self.out, self.hin, self.win, self.k, self.cf, self.out_w = name, inp, out, hin, win, k, cf, out_w
        self.hout, self.wout = hin - k + 1, win - k + 1
        self.cap_words = ((k - 1) * win + k + slack_px) * cf
        self.total_in = hin * win * cf
        self.frames, self.f = frames, 0
        self.recv = self.u = self.e = 0

    def _base_last(self):
        oy, ox = divmod(self.u, self.wout)
        return oy * self.win + ox, (oy + self.k - 1) * self.win + ox + self.k - 1

    def step(self, t: int) -> int:
        progressed = blocked = False
        done = self.f >= self.frames
        base = self._base_last()[0] if not done else self.hin * self.win
        if not done and self.recv < self.total_in and self.inp.q and self.recv - base * self.cf < self.cap_words:
            q, _ = self.inp.q.popleft()
            assert q == self.recv // self.cf, f"{self.name}: input pixel {q}, expected {self.recv // self.cf}"
            self.recv += 1
            progressed = True
        if not done:
            base, last = self._base_last()
            if self.recv >= (last + 1) * self.cf:
                if self.out.space():
                    self.out.push((self.f * self.hout * self.wout + self.u, self.e))
                    self.e += 1
                    progressed = True
                    if self.e == self.out_w:
                        self.e, self.u = 0, self.u + 1
                        if self.u == self.hout * self.wout:
                            self.u, self.recv, self.f = 0, 0, self.f + 1
                else:
                    blocked = True
        if progressed:
            return BUSY
        if blocked:
            return BLOCKED
        return IDLE if self.f >= self.frames else STARVED


def simulate_up(
    r, inject_interval: float = 0, skip_depth: int | None = None, main_depth: int | None = None, fifo_depth: int = 2,
    fifo_depths: dict | None = None, frames: int = 1, elastic_map: dict | None = None, max_cycles: int | None = None,
    swg_slack_px: int = 1, pix_simd: int | None = None,
) -> SimResult:
    """skip_depth: the 'skip FIFO' on the EXT branch end, main_depth: 'FIFO main' on the main branch end (words; None = r's value,
    UNBOUNDED = measure). inject_interval: cycles between INPUT pixels.
    pix_simd: SIMD of the FMPadding_Pixel node (None = the SWG's SIMD, what the bridge sets; 1 = FINN's default when nothing sets it: cmid folds per pixel, a DWC widens it to the SWG's SIMD)."""
    p = r.params
    H, W, Ho, Wo = p["height"], p["width"], 2 * p["height"], 2 * p["width"]
    cin, cmid, cout = p["cin"], p["cmid"], p["cout"]
    conv = p["skip_conv"]
    n_in, n_out = H * W * frames, Ho * Wo * frames
    n = {x.name: x for x in r.nodes}
    order: list = []
    fifos: dict = {}
    if skip_depth is None:
        skip_depth = r.skip_fifo.depth_words
    if main_depth is None:
        main_depth = r.params.get("main_fifo_words", 2)

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

    dup_w = cin // n["Dup"].pe
    f_src = fifo("src->Dup")
    order.append(Source("Source", f_src, n_in, dup_w, inject_interval))
    f_dm, f_de = fifo("Dup->main"), fifo("Dup->ext")
    order.append(StreamNode("Dup", [f_src], [f_dm, f_de], dup_w, dup_w, n_in))

    # ---- main branch (code name `main`): proj -> upsample -> [3x3 conv]
    sf_p, nf_p, thr_p_w = cin // n["MVAU_p"].simd, cout // n["MVAU_p"].pe, cout // n["Thr_p"].pe
    f = link("Dup", dup_w, "MVAU_p", sf_p, f_dm, n_in)
    f_o = fifo("MVAU_p->out")
    order.append(MvauNode("MVAU_p", f, f_o, sf_p, nf_p, n_in))
    f = link("MVAU_p", nf_p, "Thr_p", thr_p_w, f_o, n_in)
    f_o = fifo("Thr_p->out")
    order.append(StreamNode("Thr_p", [f], [f_o], thr_p_w, thr_p_w, n_in))
    f = link("Thr_p", thr_p_w, "UpNN", 1, f_o, n_in)
    f_up = fifo("UpNN->out")
    order.append(UpNNNode("UpNN", f, f_up, H, W, frames))
    f_main = fifo("FIFO main", main_depth)
    if conv:
        cf_k = cout // n["SWG_k"].simd
        sf_k, nf_k, thr_k_w = 9 * cout // n["MVAU_k"].simd, cout // n["MVAU_k"].pe, cout // n["Thr_k"].pe
        f = link("UpNN", 1, "FMPad_k", cf_k, f_up, n_out)
        f_o = fifo("FMPad_k->out")
        order.append(FmPadNode("FMPad_k", f, f_o, Ho, Wo, 1, cf_k, frames))
        f_o2 = fifo("SWG_k->out")
        order.append(SwgNode("SWG_k", f_o, f_o2, Ho, Wo, 3, cf_k, sf_k, swg_slack_px, frames))
        f = link("SWG_k", sf_k, "MVAU_k", sf_k, f_o2, n_out)
        f_o = fifo("MVAU_k->out")
        order.append(MvauNode("MVAU_k", f, f_o, sf_k, nf_k, n_out))
        f = link("MVAU_k", nf_k, "Thr_k", thr_k_w, f_o, n_out)
        order.append(StreamNode("Thr_k", [f], [f_main], thr_k_w, thr_k_w, n_out))
        main_w = thr_k_w
    else:
        thr_s_w = cout // n["Thr_s"].pe
        f = link("UpNN", 1, "Thr_s", thr_s_w, f_up, n_out)
        order.append(StreamNode("Thr_s", [f], [f_main], thr_s_w, thr_s_w, n_out))
        main_w = thr_s_w

    # ---- ext branch: reduce -> transposed conv (zero insertion + 2x2) -> expand
    sf_r, nf_r, thr_r_w = cin // n["MVAU_r"].simd, cmid // n["MVAU_r"].pe, cmid // n["Thr_r"].pe
    f = link("Dup", dup_w, "MVAU_r", sf_r, f_de, n_in)
    f_o = fifo("MVAU_r->out")
    order.append(MvauNode("MVAU_r", f, f_o, sf_r, nf_r, n_in))
    f = link("MVAU_r", nf_r, "Thr_r", thr_r_w, f_o, n_in)
    f_o = fifo("Thr_r->out")
    order.append(StreamNode("Thr_r", [f], [f_o], thr_r_w, thr_r_w, n_in))
    cf_u = cmid // n["SWG_u"].simd
    sf_u, nf_u, thr_u_w = 4 * cmid // n["MVAU_u"].simd, cmid // n["MVAU_u"].pe, cmid // n["Thr_u"].pe
    cf_p = cf_u if pix_simd is None else cmid // pix_simd
    f = link("Thr_r", thr_r_w, "FMPadPix", cf_p, f_o, n_in)
    f_o = fifo("FMPadPix->out")
    order.append(FmPadPixelNode("FMPadPix", f, f_o, H, W, cf_p, frames, edge_pad=False))
    f_o = link("FMPadPix", cf_p, "FMPad_u", cf_u, f_o, (2 * H - 1) * (2 * W - 1) * frames)
    f_pad = fifo("FMPad_u->out")
    order.append(FmPadNode("FMPad_u", f_o, f_pad, 2 * H - 1, 2 * W - 1, 1, cf_u, frames))
    f_o2 = fifo("SWG_u->out")
    par_u = n["MVAU_u"].simd > cmid                         # parallel_window: the SWG emits one window (cf_u words) per output pixel, a DWC splits it into sf_u words
    order.append(SwgGenNode("SWG_u", f_pad, f_o2, 2 * H + 1, 2 * W + 1, 2, cf_u, cf_u if par_u else sf_u, swg_slack_px, frames))
    f = link("SWG_u", cf_u if par_u else sf_u, "MVAU_u", sf_u, f_o2, n_out)
    f_o = fifo("MVAU_u->out")
    order.append(MvauNode("MVAU_u", f, f_o, sf_u, nf_u, n_out))
    f = link("MVAU_u", nf_u, "Thr_u", thr_u_w, f_o, n_out)
    f_o = fifo("Thr_u->out")
    order.append(StreamNode("Thr_u", [f], [f_o], thr_u_w, thr_u_w, n_out))
    sf_e, nf_e, thr_e_w = cmid // n["MVAU_e"].simd, cout // n["MVAU_e"].pe, cout // n["Thr_e"].pe
    f = link("Thr_u", thr_u_w, "MVAU_e", sf_e, f_o, n_out)
    f_o = fifo("MVAU_e->out")
    order.append(MvauNode("MVAU_e", f, f_o, sf_e, nf_e, n_out))
    f = link("MVAU_e", nf_e, "Thr_e", thr_e_w, f_o, n_out)
    f_skip = fifo("skip FIFO", skip_depth)
    order.append(StreamNode("Thr_e", [f], [f_skip], thr_e_w, thr_e_w, n_out))

    # ---- join
    add_w, thr_out_w = cout // n["Add"].pe, cout // n["Thr_out"].pe
    f_a = link("FIFOmain", main_w, "Add", add_w, f_main, n_out)
    f_b = link("skipFIFO", thr_e_w, "Add", add_w, f_skip, n_out)
    f_add = fifo("Add->out")
    order.append(StreamNode("Add", [f_a, f_b], [f_add], add_w, add_w, n_out))
    f = link("Add", add_w, "Thr_out", thr_out_w, f_add, n_out)
    f_out = fifo("Thr_out->out")
    order.append(StreamNode("Thr_out", [f], [f_out], thr_out_w, thr_out_w, n_out))
    sink = Sink("Sink", f_out, thr_out_w)
    order.append(sink)

    return run_network(
        r, order, fifos, sink, n_out, inject_interval, skip_depth, fifo_depth, None, fifo_depths, max_cycles,
        px_per_frame=(Ho * Wo if frames > 1 else None), elastic_map=elastic_map,
    )
