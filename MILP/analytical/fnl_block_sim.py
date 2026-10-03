"""Cycle-level token simulation of the ENet FINAL deconvolution (companion of fnl_block.py): the network's last layer
qnn.QuantConvTranspose2d(c5, out_channels, kernel 2, stride 2, bias=True with Int32Bias) of LayerQuantEnetFINN, lowered by FINN's
InferPixelPaddingDeconv to zero insertion + a 2x2 window + MVAU, followed by the integer bias add (ChannelwiseOp):

    FMPadPix -> SWG_u (2x2) -> MVAU_f (4*Cin -> Cout) -> Bias

Single path, no join: only ordinary FIFOs. Input H x W pixels (FmPadPix input side), output 2H x 2W (everything after); the sink counts OUTPUT pixels.
Node types come from up_bottleneck_sim (FmPadPixelNode, SwgGenNode) and bottleneck_sim.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bottleneck_sim import UNBOUNDED, Fifo, MvauNode, Sink, Source, SimResult, StreamNode, run_network  # noqa: E402,F401
from up_bottleneck_sim import FmPadPixelNode, SwgGenNode  # noqa: E402


def simulate_fnl(
    r, inject_interval: float = 0, fifo_depth: int = 2, fifo_depths: dict | None = None, frames: int = 1, elastic_map: dict | None = None,
    max_cycles: int | None = None, swg_slack_px: int = 1,
) -> SimResult:
    """inject_interval: cycles between INPUT pixels (0 = saturated)."""
    p = r.params
    H, W, Ho, Wo = p["height"], p["width"], 2 * p["height"], 2 * p["width"]
    cin, cout = p["cin"], p["cout"]
    n_in, n_out = H * W * frames, Ho * Wo * frames
    n = {x.name: x for x in r.nodes}
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

    cf = cin // n["SWG_u"].simd
    sf, nf = 4 * cin // n["MVAU_f"].simd, cout // n["MVAU_f"].pe
    bias_w = cout // n["Bias"].pe if "Bias" in n else nf
    f_src = fifo("src->FMPadPix")
    order.append(Source("Source", f_src, n_in, cf, inject_interval))
    f_o = fifo("FMPadPix->out")
    order.append(FmPadPixelNode("FMPadPix", f_src, f_o, H, W, cf, frames))
    f_o2 = fifo("SWG_u->out")
    order.append(SwgGenNode("SWG_u", f_o, f_o2, Ho + 1, Wo + 1, 2, cf, sf, swg_slack_px, frames))
    f = link("SWG_u", sf, "MVAU_f", sf, f_o2, n_out)
    f_o = fifo("MVAU_f->out")
    order.append(MvauNode("MVAU_f", f, f_o, sf, nf, n_out))
    if "Bias" in n:
        f = link("MVAU_f", nf, "Bias", bias_w, f_o, n_out)
        f_out = fifo("Bias->out")
        order.append(StreamNode("Bias", [f], [f_out], bias_w, bias_w, n_out))
    else:
        f_out = f_o
    sink = Sink("Sink", f_out, bias_w)
    order.append(sink)
    return run_network(r, order, fifos, sink, n_out, inject_interval, 0, fifo_depth, None, fifo_depths, max_cycles,
                       px_per_frame=(Ho * Wo if frames > 1 else None), elastic_map=elastic_map)
