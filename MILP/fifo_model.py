"""FIFO and DWC estimates for the MILP (finn_milp.py --model-fifos), derived from the analytical block models (MILP/analytical/).

What the analytical models showed (see MILP/analytical/analytical.md and the S12-256 artifact):
  * The memory that matters is a handful of FIFOs per block: the SKIP FIFO of every residual diamond (depth ~ n_fill + a few pixels, n_fill = pad*W + pad + 1 of the diamond's
    windowed conv) and the PREFETCH FIFO in front of every padded conv's FMPadding ((pad*W + pad + 1)*cf + 2 words). Their size in BITS is almost fixed by the geometry
    (dilation, width, channels); folding only changes the stream width (-> BRAM aspect / pow2 rounding) and a few pixels of latency. Every other FIFO of a verified block
    is <= 2 deep, which FINN's RemoveShallowFIFOs deletes. Inter-block FIFOs are fixed at depth 2 (priced 0).
  * A StreamingDataWidthConverter sits on every edge whose stream widths differ: a function of the PE / SIMD choice of the two neighbouring nodes.
Both are expressed here as terms LINEAR in the solver's one-hot fold variables z:
  * skip / prefetch / internal DWCs (MVAU -> threshold, parallel-window SWG -> MVAU) depend on ONE node's option -> a per-option constant added to that option's LUT / BRAM / URAM;
  * an inter-node DWC depends on two neighbours: width-class indicators u_P[w] (sum of z over P's options with output width w) and u_C[w'], pair variables
    m[w, w'] >= u_P[w] + u_C[w'] - 1 (continuous, minimised by the budget), priced with fcm.dwc_cost.
Simulation-only FIFOs (down feed into SWG_r, up-block `FIFO main`, init `FIFO main`) have no closed form and are NOT priced in the solve: MILP/analytical's verify_with_sim
sizes them after the fact (net_fold.py). Single candidate bit width only (every stream is `bits` wide), like the uniform S12 runs.
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "analytical"))
import finn_cost_model as fcm  # noqa: E402
from bottleneck import fifo_memory  # noqa: E402
from layer_topology import find_fork_join_diamonds  # noqa: E402

SKIP_DELTA_MIN_PX = 4         # delta = verified skip depth - n_fill, in pixels: 2..11 px in the 25 verified residual blocks (2-3% of n_fill, 8.5% at 128 px wide, 6% at 64 px / d=1);
SKIP_DELTA_FRAC = 0.085       # the bound max(4, 8.5% of n_fill) covers all of them (it is the fold-dependent term ceil(R / T) of bottleneck.py, bilinear in the fold)
DN_SKIP_MARGIN = 1.10         # the downsampling block's analytic skip depth is 8-10% below the simulated need


def n_fill_px(conv) -> int:
    """Real pixels a padded conv's sliding window needs before its first window exists: pad*W + pad + 1 (W = the conv's input width)."""
    return conv.ph * conv.win + conv.pw + 1


def skip_delta_px(n_fill: int) -> int:
    return max(SKIP_DELTA_MIN_PX, math.ceil(SKIP_DELTA_FRAC * n_fill))


def skip_depth_words(conv, cout: int, pe: int, downsampling: bool = False) -> int:
    """Skip FIFO of a residual diamond whose long branch holds the windowed `conv`: (n_fill + skip_delta_px) pixels of cout/pe words
    (the downsampling block's analytic estimate is 8-10% low -> DN_SKIP_MARGIN)."""
    px = n_fill_px(conv) + skip_delta_px(n_fill_px(conv))
    if downsampling:
        px = math.ceil(px * DN_SKIP_MARGIN)
    return px * (cout // pe)


def prefetch_depth_words(conv, simd_swu: int) -> int:
    """FIFO in front of the conv's FMPadding: the next frame's first window must be buffered ((pad*W + pad + 1)*cf + 2 words, cf = cin / simd_swu)."""
    return n_fill_px(conv) * (conv.cin // simd_swu) + 2


def _stage(name: str, geom: dict, extras: dict) -> str:
    return geom[name].stage if name in geom else extras[name].geom.stage


class FifoModel:
    def __init__(self, geometries, extra_nodes, dataflow_map, z, layer_costs, bits: int):
        self.geom = {g.name: g for g in geometries}
        self.extras = {n.geom.name: n for n in extra_nodes}
        self.dmap, self.A = dataflow_map, bits
        self.kinds = {n.geom.name: n.kind for n in extra_nodes}
        self.zvar = z
        self.options: dict[str, list[tuple]] = {}
        for key, var in z.items():
            self.options.setdefault(key[0], []).append((key, var))
        self.cost = layer_costs
        self.const: dict[tuple, dict] = {}            # key -> {lut, bram18, uram18} added to that option (skip / prefetch / internal DWCs)
        self.items: dict[tuple, list] = {}            # key -> descriptions of the FIFOs / DWCs it brings (for the output lists)
        self.skip_nodes: dict[str, dict] = {}         # skip FIFO site: node name -> {join, n_fill_px, kind}
        self._acc_final = None
        self._build_constants()
        self.edge_terms = []                          # filled by add_edge_terms

    # ------------------------------------------------------------------ port widths
    def _is_layer(self, name: str) -> bool:
        return name in self.geom

    def ports(self, key: tuple, pred: str | None = None) -> dict:
        """in_w / out_w of the node's streams (bits) for option `key`, plus `internal` DWCs [(label, w_in, w_out)] inside the node's own FINN subgraph."""
        name, A = key[0], self.A
        pe, simd = key[1], key[2]
        if self._is_layer(name):
            g, c = self.geom[name], self.cost[key]
            if g.op_type == "MaxPool2d":
                return dict(in_w=g.cin * A, out_w=g.cin * A, internal=[])
            internal = []
            if g.kh * g.kw > 1 and simd > g.cin:                                   # parallel_window: one KxK window per SWG word, a DWC narrows it to the MVAU's SIMD
                internal.append(("swg->mvau", g.kh * g.kw * c["simd_swu"] * A, simd * A))
            if c["thr_pe"]:
                internal.append(("mvau->thr", pe * c["acc_bits"], c["thr_pe"] * c["acc_bits"]))
            out_w = c["thr_pe"] * A if c["thr_pe"] else pe * c["acc_bits"]
            return dict(in_w=c["simd_swu"] * A, out_w=out_w, internal=internal)
        kind, g = self.extras[name].kind, self.extras[name].geom
        if kind in ("skip_quant", "input_quant", "act", "pool_quant"):
            acc_pad = self._acc_of_pad(pred)
            return dict(in_w=pe * (acc_pad if acc_pad else A), out_w=pe * A, internal=[])
        if kind == "residual_add":
            return dict(in_w=pe * (A + 1), out_w=pe * A, internal=[])
        if kind == "out_act":
            return dict(in_w=pe * A, out_w=pe * A, internal=[])
        if kind == "add":
            return dict(in_w=pe * A, out_w=pe * (A + 1), internal=[])
        if kind == "dup":
            return dict(in_w=pe * A, out_w=pe * A, internal=[])
        if kind in ("concat", "upsample"):
            return dict(in_w=g.cout * A, out_w=g.cout * A, internal=[])
        if kind == "pad_mvau":
            return dict(in_w=simd * A, out_w=pe * self.cost[key]["acc_bits"], internal=[])
        if kind == "argmax":
            return dict(in_w=pe * self._final_acc(), out_w=8, internal=[])
        raise ValueError(f"fifo_model: no port widths for extra node kind {kind!r}")

    def _acc_of_pad(self, pred: str | None) -> int:
        if pred is not None and pred in self.extras and self.extras[pred].kind == "pad_mvau":
            return self.cost[self.options[pred][0][0]]["acc_bits"]
        return 0

    def _final_acc(self) -> int:
        if self._acc_final is None:
            last = next(n for n in reversed(list(self.geom)) if self.geom[n].op_type != "MaxPool2d")
            self._acc_final = self.cost[self.options[last][0][0]]["acc_bits"]
        return self._acc_final

    def in_w_for_edge(self, key: tuple, pred: str) -> int:
        name = key[0]
        if name in self.extras and self.extras[name].kind == "concat":              # concat reads each predecessor's channels at once
            g = self.geom.get(pred) or self.extras[pred].geom
            return g.cout * self.A if pred in self.geom else self.extras[pred].geom.cout * self.A
        return self.ports(key, pred)["in_w"]

    # ------------------------------------------------------------------ per-option constants: skip, prefetch, internal DWCs
    def _add_const(self, key, lut=0.0, bram=0.0, uram=0.0, item=None):
        c = self.const.setdefault(key, dict(lut=0.0, bram18=0.0, uram18=0.0))
        c["lut"] += lut
        c["bram18"] += bram
        c["uram18"] += uram
        if item is not None:
            self.items.setdefault(key, []).append(item)

    def _fifo_item(self, key, label, width, depth, stage, is_skip=False, producer=None, consumer=None):
        m = fifo_memory(width, depth)
        self._add_const(key, m["lut"], m["bram18"], m["uram"], dict(
            kind="fifo", name=label, stage=stage, producer=producer, consumer=consumer, width_bits=int(width), depth=int(depth), mem=m["mem"],
            depth_alloc=int(m["depth_alloc"]), lut=int(m["lut"]), mem_bram18=int(m["bram18"]), mem_uram18=int(m["uram"]), is_skip=is_skip))

    def _build_constants(self) -> None:
        A = self.A
        # internal DWCs of every option
        for name, opts in self.options.items():
            for key, _ in opts:
                for label, w1, w2 in self.ports(key)["internal"]:
                    if w1 != w2:
                        lut = fcm.dwc_cost(w1, w2)["total_lut"]
                        self._add_const(key, lut=lut, item=dict(kind="dwc", name=f"{name}.{label}", stage=_stage(name, self.geom, self.extras), in_width=int(w1),
                                                                out_width=int(w2), lut=float(lut)))
        # prefetch FIFO in front of every padded windowed conv's FMPadding
        for name, g in self.geom.items():
            if g.op_type == "MaxPool2d" or g.kh * g.kw == 1 or not (g.ph or g.pw):
                continue
            for key, _ in self.options[name]:
                c = self.cost[key]
                self._fifo_item(key, f"{name}.prefetch", c["simd_swu"] * A, prefetch_depth_words(g, c["simd_swu"]), g.stage, producer=f"{name}", consumer=f"{name}.fmpad")
        # skip FIFO of every residual diamond (join kind add)
        for d in find_fork_join_diamonds(self.dmap):
            if self.kinds.get(d["join"]) != "add":
                continue
            conv_branch = None
            for label, other in (("branch_a", "branch_b"), ("branch_b", "branch_a")):
                convs = [n for n in d[label] if n in self.geom and self.geom[n].kh * self.geom[n].kw > 1 and (self.geom[n].ph or self.geom[n].pw)]
                if convs:
                    conv_branch, short, conv = label, d[other], self.geom[convs[0]]
                    break
            if conv_branch is None or not short or short[0] not in self.extras:
                continue                                                   # up blocks: the long branch is a transposed conv -> simulation-only FIFO
            node = short[0]                                                # the node right before the join on the short branch
            dn = self.extras[d["join"]].geom.stage.startswith("down")
            cout = self.extras[d["join"]].geom.cout
            self.skip_nodes[node] = dict(join=d["join"], n_fill_px=n_fill_px(conv), conv=conv.name)
            for key, _ in self.options[node]:
                pe = key[1]
                self._fifo_item(key, f"{d['join']}.skip", pe * A, skip_depth_words(conv, cout, pe, dn), self.extras[d["join"]].geom.stage, is_skip=True,
                                producer=node, consumer=d["join"])

    # ------------------------------------------------------------------ solver terms
    def add_terms(self, pulp, prob, tag: str = "fifo"):
        """Return (lut_expr, bram_expr, uram_expr): per-option constants + inter-node DWC pair terms. Creates the m[] variables on `prob`."""
        lut = [self.zvar[key] * c["lut"] for key, c in self.const.items() if c["lut"]]
        bram = [self.zvar[key] * c["bram18"] for key, c in self.const.items() if c["bram18"]]
        uram = [self.zvar[key] * c["uram18"] for key, c in self.const.items() if c["uram18"]]
        n_pairs = 0
        for consumer, preds in self.dmap.items():
            if consumer not in self.options:
                continue
            for pred in preds:
                if pred not in self.options:
                    continue
                out_cls = self._classes(pred, lambda k: self.ports(k)["out_w"])
                in_cls = self._classes(consumer, lambda k, p=pred: self.in_w_for_edge(k, p))
                for w1, zs1 in out_cls.items():
                    for w2, zs2 in in_cls.items():
                        if w1 == w2:
                            continue
                        cost = fcm.dwc_cost(w1, w2)["total_lut"]
                        if len(out_cls) == 1 and len(in_cls) == 1:
                            lut.append(cost)
                            self.edge_terms.append((pred, consumer, w1, w2, None))
                            continue
                        m = pulp.LpVariable(f"{tag}_dwc_{n_pairs}", lowBound=0, upBound=1)
                        n_pairs += 1
                        prob += m >= pulp.lpSum(zs1) + pulp.lpSum(zs2) - 1, f"{tag}_dwc_def_{n_pairs}"
                        lut.append(m * cost)
                        self.edge_terms.append((pred, consumer, w1, w2, m))
        self.n_pair_vars = n_pairs
        return pulp.lpSum(lut), pulp.lpSum(bram), pulp.lpSum(uram)

    def _classes(self, name: str, width_of) -> dict:
        out: dict[int, list] = {}
        for key, var in self.options[name]:
            out.setdefault(int(width_of(key)), []).append(var)
        return out

    # ------------------------------------------------------------------ outputs
    def describe(self, pulp, chosen: dict) -> dict:
        """After the solve. chosen: node name -> chosen z key. Returns the FIFO / DWC lists and the totals."""
        fifos, dwcs = [], []
        tot = dict(lut=0.0, bram18=0.0, uram18=0.0, dwc_lut=0.0)
        for name, key in chosen.items():
            for it in self.items.get(key, []):
                if it["kind"] == "fifo":
                    fifos.append({k: v for k, v in it.items() if k != "kind"})
                    tot["lut"] += it["lut"]
                    tot["bram18"] += it["mem_bram18"]
                    tot["uram18"] += it["mem_uram18"]
                else:
                    dwcs.append({k: v for k, v in it.items() if k != "kind"})
                    tot["dwc_lut"] += it["lut"]
        for pred, consumer, w1, w2, _m in self.edge_terms:          # the chosen widths decide, whatever value the relaxed pair variable took
            if not (self._chosen_width(chosen, pred, "out") == w1 and self._chosen_width(chosen, consumer, "in", pred) == w2):
                continue
            lut = fcm.dwc_cost(w1, w2)["total_lut"]
            dwcs.append(dict(name=f"{pred}->{consumer}", stage=_stage(consumer, self.geom, self.extras), in_width=int(w1), out_width=int(w2), lut=float(lut)))
            tot["dwc_lut"] += lut
        return dict(fifos=fifos, dwcs=dwcs, totals=tot)

    def _chosen_width(self, chosen, name, side, pred=None):
        key = chosen[name]
        return self.in_w_for_edge(key, pred) if side == "in" else self.ports(key)["out_w"]

    def inter_block_fifos(self, chosen: dict, depth: int = 2) -> list[dict]:
        """One depth-`depth` FIFO per dataflow edge leaving a block (the same list net_fold.py writes)."""
        out = []
        for consumer, preds in self.dmap.items():
            if consumer not in chosen:
                continue
            for p in preds:
                if p not in chosen:
                    continue
                sp, sc = _stage(p, self.geom, self.extras), _stage(consumer, self.geom, self.extras)
                if sp == sc:
                    continue
                w = self.ports(chosen[p])["out_w"]
                m = fifo_memory(w, depth)
                out.append(dict(name=f"{p}->{consumer}", producer=p, consumer=consumer, src_block=sp, dst_block=sc, width_bits=int(w), mem=m["mem"], depth=depth,
                                depth_alloc=int(m["depth_alloc"]), bram18=int(m["bram18"]), lut=int(m["lut"]), uram18=int(m["uram"]), sizing="fixed"))
        return out
