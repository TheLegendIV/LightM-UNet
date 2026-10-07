"""Inter-block FIFO sizing by chained simulation (companion of net_fold.py).

Every block model (bottleneck / dn / up / int / fnl) is realised and verified on its own: its intra-block FIFOs, DWCs and window buffers are
sized by `verify_with_sim` and are treated here as a BLACK BOX (their verified depths are replayed unchanged). What is left to decide is the FIFO
at the OUTPUT of each block (the edge to the next block). It is found by walking the network:

    for every pair (block i, block i+1):  A saturated source -> block i -> FIFO(D) [-> DWC] -> block i+1 -> sink
    smallest D (in words of the producer's stream) whose last-frame period is <= the global target cycles per frame.

The block networks are built by the blocks' own `simulate_*` functions (bottleneck_sim.CAPTURE makes run_network hand the built network back instead
of running it), glued at the interface and stepped by `run_chain`. The same glue chains the WHOLE net for a closing check. Depth <= 2 is what the
nodes' own handshakes already give; FINN's RemoveShallowFIFOs deletes such FIFOs, so 2 is the floor.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bottleneck_sim as bs  # noqa: E402
from bottleneck_sim import BUSY, CAPTURE, Sink, Source, StreamNode  # noqa: E402

MIN_DEPTH = 2
DEFAULT_FRAMES = 3                 # steady period = the LAST frame's period (warm-up frames excluded)
REFINE_REL = 1 / 8                 # stop bisecting when hi - lo <= hi/8 (BRAM sizes only change at powers of two; SRL cost is linear)


def _sim_of(kind: str):
    if kind == "reg":
        return bs.simulate
    if kind == "dn":
        from dn_bottleneck_sim import simulate_dn
        return simulate_dn
    if kind == "up":
        from up_bottleneck_sim import simulate_up
        return simulate_up
    if kind == "init":
        from int_bottleneck_sim import simulate_int
        return simulate_int
    from fnl_block_sim import simulate_fnl
    return simulate_fnl


def verified_depths(r) -> dict:
    """The intra-block FIFO depths verify_with_sim settled on (every FIFO except the two that touch the block's Source / Sink)."""
    return {n: f["depth"] for n, f in r.fifo_graph["fifos"].items() if f["producer"] != "Source" and f["consumer"] != "Sink"}


def capture_block(kind: str, r, frames: int = DEFAULT_FRAMES) -> dict:
    """Build (not run) the block's simulation network with its verified intra-block depths, saturated source, `frames` back-to-back frames."""
    CAPTURE["out"] = out = []
    try:
        _sim_of(kind)(r, inject_interval=0, fifo_depths=verified_depths(r), frames=frames)
    finally:
        CAPTURE["out"] = None
    b = out[0]
    b["source"], b["frames"] = b["order"][0], frames
    assert isinstance(b["source"], Source) and isinstance(b["sink"], Sink)
    return b


def _swap(node, old, new) -> None:
    for a in ("inp", "out"):
        if getattr(node, a, None) is old:
            setattr(node, a, new)
    for a in ("ins", "outs"):
        lst = getattr(node, a, None)
        if lst:
            for i, f in enumerate(lst):
                if f is old:
                    lst[i] = new


def compose(blocks: list, depths: list) -> dict:
    """Glue captured blocks into one chain. depths[i] = depth (words of block i's output stream) of the FIFO between block i and i+1.
    Returns order (topological), the chain's sink, total output pixels and the interface FIFOs."""
    assert len(depths) == len(blocks) - 1
    order, iface = [], []
    for i, b in enumerate(blocks):
        nodes = list(b["order"])
        if i > 0:
            nodes = nodes[1:]               # this block's Source is replaced by the previous block's output FIFO
        if i < len(blocks) - 1:
            nodes = nodes[:-1]              # this block's Sink is replaced by the interface FIFO
        if i > 0:
            prev, src = blocks[i - 1], b["source"]
            f_out, f_in = prev["sink"].inp, src.out
            f_out.depth, f_out.name = depths[i - 1], f"iface{i - 1}"
            assert prev["N"] == src.n, f"pixel count mismatch at interface {i - 1}: {prev['N']} vs {src.n}"
            if prev["sink"].words == src.words:
                for n in nodes:
                    _swap(n, f_in, f_out)
            else:                           # FINN: producer FIFO -> DWC -> (2-deep) consumer FIFO
                f_in.depth = MIN_DEPTH
                order.append(StreamNode(f"DWC(iface{i - 1})", [f_out], [f_in], prev["sink"].words, src.words, src.n))
            iface.append(f_out)
        order.extend(nodes)
    last = blocks[-1]
    return dict(order=order, sink=last["sink"], N=last["N"], px_per_frame=last["N"] // last["frames"], iface=iface)


def chain_fifos(net: dict) -> list:
    """Every distinct Fifo object of a composed chain (object identity: the intra-block FIFO names repeat across blocks)."""
    seen, out = set(), []
    for node in net["order"]:
        for attr in ("inp", "out", "ins", "outs"):
            v = getattr(node, attr, None)
            for f in (v if isinstance(v, (list, tuple)) else [v]):
                if isinstance(f, bs.Fifo) and id(f) not in seen:
                    seen.add(id(f))
                    out.append(f)
    return out


def run_chain(net: dict, max_cycles: int, stall: int = 5000) -> dict:
    """Step the chain until the sink has N pixels. Returns deadlock flag, per-frame periods (cycles), interface FIFO occupancies, and the Fifo objects that were
    full when the run ended (`full`: the blocked chain of a deadlock) or reached their depth at any time (`saturated`)."""
    order, sink, N, ppf = net["order"], net["sink"], net["N"], net["px_per_frame"]
    rev = list(reversed(order))
    t = last = 0
    deadlock = False
    while len(sink.done) < N:
        prog = False
        for node in rev:
            if node.step(t) == BUSY:
                prog = True
        if prog:
            last = t
        elif t - last > stall:
            deadlock = True
            break
        t += 1
        if t >= max_cycles:
            deadlock = True
            break
    done = sink.done
    ends = [done[(k + 1) * ppf - 1] for k in range(len(done) // ppf)]
    periods = [ends[k] - ends[k - 1] for k in range(1, len(ends))]
    fifos = chain_fifos(net)
    return dict(deadlock=deadlock, cycles=t, periods=periods, period=(periods[-1] if periods else float("inf")),
                first_out=(done[0] if done else -1), iface_max=[f.max_occ for f in net["iface"]],
                full=[f for f in fifos if len(f.q) >= f.depth], saturated=[f for f in fifos if f.max_occ >= f.depth])


def run_pair(a: tuple, b: tuple, depth: int, F: int, frames: int = DEFAULT_FRAMES) -> dict:
    """a, b = (kind, verified block result). One saturated-input run of block a + FIFO(depth) + block b."""
    net = compose([capture_block(*a, frames), capture_block(*b, frames)], [depth])
    return run_chain(net, max_cycles=int(frames * F * 3) + 100000)


def find_interface_depth(a: tuple, b: tuple, F: int, slack: float = 0.0, frames: int = DEFAULT_FRAMES) -> dict:
    """Smallest FIFO depth (words) between block a and block b with last-frame period <= F * (1 + slack).
    Doubling from MIN_DEPTH, then bisection down to REFINE_REL. A frame-sized FIFO always decouples the blocks completely; if even that misses the target the
    pair is limited by the blocks themselves (reported, depth = the occupancy the unbounded run needs)."""
    lim = F * (1 + slack)
    tests = {}

    def ok(d: int) -> bool:
        res = tests[d] = run_pair(a, b, d, F, frames)
        return (not res["deadlock"]) and res["period"] <= lim

    cap = _frame_words(a, frames)
    d = MIN_DEPTH
    lo = None
    while not ok(d):
        lo = d
        if d >= cap:
            break
        d = min(cap, d * 2)
    status = "ok"
    if tests[d]["deadlock"] or tests[d]["period"] > lim:                  # limited by the blocks, not by the FIFO
        status = "block_limited"
        d = max(MIN_DEPTH, tests[d]["iface_max"][0])
        ok(d)
    elif lo is not None:
        hi = d
        while hi - lo > max(1, hi * REFINE_REL):
            mid = (lo + hi) // 2
            if ok(mid):
                hi = mid
            else:
                lo = mid
        d = hi
    res = tests[d]
    return dict(depth=int(d), status=status, period=res["period"], max_occ=res["iface_max"][0], sims=len(tests),
                tested={int(k): (None if v["deadlock"] else int(v["period"])) for k, v in sorted(tests.items())})


def _frame_words(a: tuple, frames: int) -> int:
    b = capture_block(*a, 1)
    return b["N"] * b["sink"].words + MIN_DEPTH


def _pair_task(args):
    i, a, b, F, slack, frames = args
    return i, find_interface_depth(a, b, F, slack, frames)


def size_interfaces(blocks: list, F: int, slack: float = 0.0, frames: int = DEFAULT_FRAMES, workers: int = 1, log=print) -> list:
    """Walk the net: blocks = [(stage, kind, verified result)] in dataflow order; returns one dict per interface (len(blocks) - 1)."""
    tasks = [(i, (blocks[i][1], blocks[i][2]), (blocks[i + 1][1], blocks[i + 1][2]), F, slack, frames) for i in range(len(blocks) - 1)]
    out = [None] * len(tasks)
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for i, res in pool.imap_unordered(_pair_task, tasks):
                out[i] = res
                log(f"  {blocks[i][0]} -> {blocks[i + 1][0]}: depth {res['depth']} ({res['status']}, period {res['period']}, {res['sims']} sims)")
    else:
        for t in tasks:
            i, res = _pair_task(t)
            out[i] = res
            log(f"  {blocks[i][0]} -> {blocks[i + 1][0]}: depth {res['depth']} ({res['status']}, period {res['period']}, {res['sims']} sims)")
    return out


def run_whole(blocks: list, depths: list, F: int, frames: int = DEFAULT_FRAMES) -> dict:
    """Closing check: every block chained with the sized interface FIFOs, saturated input, last-frame period vs F."""
    net = compose([capture_block(k, r, frames) for _, k, r in blocks], depths)
    res = run_chain(net, max_cycles=int(frames * F * 3) + 400000)
    res["target"] = F
    res["ok"] = (not res["deadlock"]) and res["period"] <= F
    return res
