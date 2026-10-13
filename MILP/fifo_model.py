"""FIFO and DWC estimates for the MILP (finn_milp.py --model-fifos), derived from the analytical block models (MILP/analytical/).

What the analytical models showed (see MILP/analytical/analytical.md and the S12-256 artifact):
  * The memory that matters is a handful of FIFOs per block: the SKIP FIFO of every residual diamond (depth ~ n_fill + a few pixels, n_fill = pad*W + pad + 1 of the diamond's
    windowed conv) and the PREFETCH FIFO in front of every padded conv's FMPadding ((pad*W + pad + 1)*cf + 2 words). Their size in BITS is almost fixed by the geometry
    (dilation, width, channels); folding only changes the stream width (-> BRAM aspect / pow2 rounding) and a few pixels of latency. Every other FIFO of a verified block
    is <= 2 deep. The real build does NOT run RemoveShallowFIFOs (step_set_fifo_depths_fixed2), so each of them stays a depth-2 SRL FIFO on its edge (~6.3 + 1.0*width LUT, 8.6k LUT in
    the bilinear-256 build): priced per edge by _build_edge_fifos, inter-block edges included.
  * A StreamingDataWidthConverter sits on every edge whose stream widths differ: a function of the PE / SIMD choice of the two neighbouring nodes.
Both are expressed here as terms LINEAR in the solver's one-hot fold variables z:
  * skip / prefetch / internal DWCs (MVAU -> threshold, parallel-window SWG -> MVAU) depend on ONE node's option -> a per-option constant added to that option's LUT / BRAM / URAM;
  * an inter-node DWC depends on two neighbours: width-class indicators u_P[w] (sum of z over P's options with output width w) and u_C[w'], pair variables
    m[w, w'] >= u_P[w] + u_C[w'] - 1 (continuous, minimised by the budget), priced with fcm.dwc_cost.
The up blocks' two join FIFOs (`skip FIFO` at the ext end = one input row, `FIFO main` = 3 input rows of the main stream; closed forms fitted to the simulation with the real upsampler kernel, see
UP_SKIP_ROWS / UP_MAIN_ROWS) are priced like the regular skip FIFO. Every intra-block FIFO of a DOWN or UP block (both ends in the same block; inter-block edges stay at 2) is BLOCK_FIFO_DEPTH = 32 words deep, not FINN's 2: those blocks' nodes are balanced within a few % of the
budget and the zero insertion / strided window / upsampler move whole rows in bursts, so 2-word FIFOs couple the phases (512x512 INT6: down1 +29 %, down2 +26 %, up4 +81 %, up5 +48 % over the MILP cycles,
on target at 32; bottleneck.BLOCK_FIFO_DEPTH). 32 words is the last depth at which fifo_memory's SRL FIFO costs the same, so the term is a per-width constant. The cycles stay the nodes'; the FIFOs that make them
true are priced. Simulation-only FIFOs without a closed form (down feed into SWG_r, init `FIFO main`, dup outputs) are NOT priced in the solve:
MILP/analytical's verify_with_sim sizes them after the fact (net_fold.py). Mixed precision works since 2026-10-11: every width uses the bits of the option that owns the FIFO (FifoModel._bits); neighbours with different bits then see a width mismatch (a small spurious DWC term, the real stream carries one dtype), and bit-less nodes use the largest candidate.
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "analytical"))
import finn_cost_model as fcm  # noqa: E402
from bottleneck import fifo_memory  # noqa: E402
from bottleneck import BLOCK_FIFO_DEPTH  # noqa: E402
from layer_topology import find_fork_join_diamonds  # noqa: E402

SKIP_DELTA_MIN_PX = 4         # delta = verified skip depth - n_fill, in pixels: 2..11 px in the 25 verified residual blocks (2-3% of n_fill, 8.5% at 128 px wide, 6% at 64 px / d=1);
SKIP_DELTA_FRAC = 0.085       # the bound max(4, 8.5% of n_fill) covers all of them (it is the fold-dependent term ceil(R / T) of bottleneck.py, bilinear in the fold)
DN_SKIP_MARGIN = 1.10         # the downsampling block's analytic skip depth is 8-10% below the simulated need
UP_SKIP_ROWS = 1.0            # up block, ext-end skip FIFO: ONE input row of the block (W px of the ext stream). Simulation with the real row-buffer UpsampleNearestNeighbour (MILP/analytical,
                              # up_bottleneck_sim.UpNNNode, 2026-10-08): paced need = W-1 px for every fold swept (up4 27-31 px of W=32, up5 58.5-63 px of W=64), final depth need + 1 px = W px exactly.
UP_MAIN_ROWS = 3.0            # up block, `FIFO main` (proj -> UpNN -> Thr_s end): bound 3 x W px of the main stream. The paced need depends on the Thr_s fold and the rate (1 px .. 89 px at W=32, 5 .. 145 px
                              # at W=64 over budgets 16..96 / 4..24 cycles per output px; max 2.75 W); the MILP's own up4 / up5 folds need 22 / 40.5 px. It is a deadlock bound, cheap in memory (one BRAM18).


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


UP_SKIP_ROWS_DW = 6.0         # up block WITH the bilinear decoder's frozen depthwise conv: its window (2 rows + 3 px of the OUTPUT image through the depthwise SWG, 16 / 4 channel folds at PE 1) delays the main branch,
                              # so the ext-end skip FIFO holds ~6 input rows: simulation (mixed 6/8, 512x512): up4 5,904 words = 5.8 rows of 64 px x 16, up5 2,604 = 5.1 rows of 128 px x 4 -> 6.0 covers both
UP_MAIN_ROWS_DW = 0.5         # ... and the main branch, now the LONG one, needs almost no join buffer: simulation 17 words (up4, 0.02 rows); half a row is a conservative bound


def up_skip_depth_words(w_in: int, cout: int, pe: int, dw: bool = False) -> int:
    """Up block, FIFO at the end of the ext branch (reduce -> transposed conv -> expand), `skip FIFO` of the analytical block: UP_SKIP_ROWS x W px of cout/pe words (W = input width of the block)."""
    return math.ceil((UP_SKIP_ROWS_DW if dw else UP_SKIP_ROWS) * w_in) * (cout // pe)


def up_main_depth_words(w_in: int, cout: int, pe: int, dw: bool = False) -> int:
    """Up block, `FIFO main` at the end of the main branch (main_proj -> UpsampleNearestNeighbour -> skip_quant): UP_MAIN_ROWS x W px of cout/pe words."""
    return math.ceil((UP_MAIN_ROWS_DW if dw else UP_MAIN_ROWS) * w_in) * (cout // pe)


def prefetch_depth_words(conv, simd_swu: int) -> int:
    """FIFO in front of the conv's FMPadding: the next frame's first window must be buffered ((pad*W + pad + 1)*cf + 2 words, cf = cin / simd_swu)."""
    return n_fill_px(conv) * (conv.cin // simd_swu) + 2


def _stage(name: str, geom: dict, extras: dict) -> str:
    return geom[name].stage if name in geom else extras[name].geom.stage


class FifoModel:
    def __init__(self, geometries, extra_nodes, dataflow_map, z, layer_costs, bits):
        self.geom = {g.name: g for g in geometries}
        self.extras = {n.geom.name: n for n in extra_nodes}
        # bits: ONE int (uniform run) or the candidate bit widths (mixed precision). Every FIFO / DWC width uses the bits of the option that owns it (self._bits(key) = the option's
        # activation bits); nodes without bits of their own (dup / concat / upsample / argmax: key bits 0) and concat's input widths use self.A = the LARGEST candidate (conservative).
        self.dmap = dataflow_map
        self.A = max([bits] if isinstance(bits, int) else list(bits))
        self.kinds = {n.geom.name: n.kind for n in extra_nodes}
        self.zvar = z
        self.options: dict[str, list[tuple]] = {}
        for key, var in z.items():
            self.options.setdefault(key[0], []).append((key, var))
        self.cost = layer_costs
        self.const: dict[tuple, dict] = {}            # key -> {lut, bram18, uram18} added to that option (skip / prefetch / internal DWCs)
        self.items: dict[tuple, list] = {}            # key -> descriptions of the FIFOs / DWCs it brings (for the output lists)
        self.skip_nodes: dict[str, dict] = {}         # skip FIFO site: node name -> {join, n_fill_px, kind}
        self.replaced_edges: set[tuple] = set()       # (producer, consumer) edges whose depth-2 FIFO is replaced by a deeper skip / prefetch FIFO
        self._acc_final = None
        self._build_constants()
        self._build_edge_fifos()
        self.edge_terms = []                          # filled by add_edge_terms

    # ------------------------------------------------------------------ port widths
    def _is_layer(self, name: str) -> bool:
        return name in self.geom

    def _bits(self, key: tuple) -> int:
        """Stream precision of an option: its own activation bits (key[6]), or the conservative maximum candidate for nodes that carry none."""
        return int(key[6]) if key[6] else self.A

    def ports(self, key: tuple, pred: str | None = None) -> dict:
        """in_w / out_w of the node's streams (bits) for option `key`, plus `internal` DWCs [(label, w_in, w_out)] inside the node's own FINN subgraph."""
        name, A = key[0], self._bits(key)
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
        if kind == "frozen_dw":                                                             # FMPadding -> depthwise SWG (SIMD = PE) -> VVAU; the threshold behind it is the skip_quant node
            c = self.cost[key]
            internal = [("swg->mvau", g.kh * g.kw * c["simd_swu"] * A, pe * simd * A)] if simd > 1 else []     # parallel_window: one KxK window per word, a DWC narrows it to the VVAU's SIMD
            return dict(in_w=c["simd_swu"] * A, out_w=pe * c["acc_bits"], internal=internal)
        if kind == "bias":                                                                  # ChannelwiseOp: INT16 -> INT17 in the real build
            return dict(in_w=pe * self._final_acc(), out_w=pe * (self._final_acc() + 1), internal=[])
        if kind == "argmax":
            acc = self._final_acc() + (1 if any(k == "bias" for k in self.kinds.values()) else 0)     # behind the bias add the logits are one bit wider
            return dict(in_w=pe * acc, out_w=8, internal=[])
        raise ValueError(f"fifo_model: no port widths for extra node kind {kind!r}")

    def _acc_of_pad(self, pred: str | None) -> int:
        if pred is not None and pred in self.extras and self.extras[pred].kind in ("pad_mvau", "frozen_dw"):
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

    def _fifo_item(self, key, label, width, depth, stage, is_skip=False, producer=None, consumer=None, role_kind="", conv=None):
        m = fifo_memory(width, depth)
        self._add_const(key, m["lut"], m["bram18"], m["uram"], dict(
            kind="fifo", name=label, stage=stage, producer_milp=producer, consumer_milp=consumer, width_bits=int(width), depth=int(depth), mem=m["mem"],
            depth_alloc=int(m["depth_alloc"]), lut=int(m["lut"]), mem_bram18=int(m["bram18"]), mem_uram18=int(m["uram"]), is_skip=is_skip,
            role_kind=role_kind, conv=conv))

    def _build_constants(self) -> None:
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
            self.replaced_edges.update((p, name) for p in self.dmap.get(name, []))
            for key, _ in self.options[name]:
                c = self.cost[key]
                self._fifo_item(key, f"{name}.prefetch", c["simd_swu"] * self._bits(key), prefetch_depth_words(g, c["simd_swu"]), g.stage, producer=f"{name}", consumer=f"{name}.fmpad",
                                 role_kind="prefetch", conv=name)
        # ... and in front of the frozen depthwise conv's FMPadding (an extra node, not a traced layer)
        for name, node in self.extras.items():
            if node.kind != "frozen_dw":
                continue
            self.replaced_edges.update((p, name) for p in self.dmap.get(name, []))
            for key, _ in self.options[name]:
                c = self.cost[key]
                self._fifo_item(key, f"{name}.prefetch", c["simd_swu"] * self._bits(key), prefetch_depth_words(node.geom, c["simd_swu"]), node.geom.stage, producer=name, consumer=f"{name}.fmpad",
                                 role_kind="prefetch_dw", conv=name)
        # skip FIFO of every residual diamond (join kind add)
        for d in find_fork_join_diamonds(self.dmap):
            if self.kinds.get(d["join"]) != "add":
                continue
            if self.extras[d["join"]].geom.stage.startswith("up"):
                self._up_block_fifos(d)                                            # the long branch is a lowered transposed conv: closed forms from the simulation (up_skip_depth_words)
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
            self.replaced_edges.add((node, d["join"]))
            for key, _ in self.options[node]:
                pe = key[1]
                self._fifo_item(key, f"{d['join']}.skip", pe * self._bits(key), skip_depth_words(conv, cout, pe, dn), self.extras[d["join"]].geom.stage, is_skip=True,
                                producer=node, consumer=d["join"], role_kind="skip")

    def _build_edge_fifos(self) -> None:
        """The real build (hardware/finn_s12_build_steps.step_set_fifo_depths_fixed2) keeps ONE depth-2 StreamingFIFO on EVERY dataflow edge (no RemoveShallowFIFOs); it is a few LUTs of SRL
        per FIFO (bottleneck.fifo_memory: 6.3 + 1.0*width), 8.6k LUT over the 440 of the bilinear-256 build. The width is the producer's output stream -> a per-option constant of the
        producer, one per outgoing edge; edges carrying a deeper skip / prefetch FIFO (self.replaced_edges) are priced by that FIFO instead."""
        for consumer, preds in self.dmap.items():
            if consumer not in self.options:
                continue
            for pred in preds:
                if pred not in self.options or (pred, consumer) in self.replaced_edges:
                    continue
                for key, _ in self.options[pred]:
                    self._edge_fifo(key, f"{pred}->{consumer}", self.ports(key)["out_w"], self._edge_depth(pred, consumer))
        self._dw_internal_fifos()
        # FIFOs INSIDE a conv's FINN subgraph (FMPadding -> SWG -> MVAU -> Thresholding: one per node output; the Thr output is the dataflow edge above), and one after every DWC
        for name, g in self.geom.items():
            if g.op_type == "MaxPool2d":
                continue
            for key, _ in self.options[name]:
                c = self.cost[key]
                A = self._bits(key)
                d_int = BLOCK_FIFO_DEPTH if self._deep_stage(_stage(name, self.geom, self.extras)) else 2      # inside a down / up block every FIFO is BLOCK_FIFO_DEPTH deep
                if g.kh * g.kw > 1:
                    if g.ph or g.pw:
                        self._edge_fifo(key, f"{name}.fmpad->swg", c["simd_swu"] * A, d_int)
                    elif g.op_type == "ConvTranspose2d":                                    # lowered: FMPadding_Pixel -> FMPadding -> SWG (the sim's FMPadPix->out / FMPad_u->out)
                        self._edge_fifo(key, f"{name}.fmpadpix->fmpad", c["simd_swu"] * A, d_int)
                        self._edge_fifo(key, f"{name}.fmpad->swg", c["simd_swu"] * A, d_int)
                    parallel = key[2] > g.cin                                                # parallel_window: the SWG emits a whole KxK window per word
                    self._edge_fifo(key, f"{name}.swg->mvau", (g.kh * g.kw if parallel else 1) * c["simd_swu"] * A, d_int)
                self._edge_fifo(key, f"{name}.mvau->thr", key[1] * c["acc_bits"], d_int)
                for label, w1, w2 in self.ports(key)["internal"]:
                    if w1 != w2:
                        self._edge_fifo(key, f"{name}.{label}.dwc_out", w2, d_int)

    @staticmethod
    def _deep_stage(stage: str) -> bool:
        return stage.startswith(("down", "up"))

    def _dw_internal_fifos(self) -> None:
        """FMPadding -> SWG -> VVAU inside the frozen depthwise conv (the VVAU -> skip_quant edge is an ordinary dataflow edge): BLOCK_FIFO_DEPTH words like every intra-block FIFO of an up block."""
        for name, node in self.extras.items():
            if node.kind != "frozen_dw":
                continue
            for key, _ in self.options[name]:
                c, A = self.cost[key], self._bits(key)
                self._edge_fifo(key, f"{name}.fmpad->swg", c["simd_swu"] * A, BLOCK_FIFO_DEPTH)
                self._edge_fifo(key, f"{name}.swg->mvau", (node.geom.kh * node.geom.kw if key[2] > 1 else 1) * c["simd_swu"] * A, BLOCK_FIFO_DEPTH)

    def _edge_depth(self, pred: str, consumer: str) -> int:
        """Depth of the FIFO on the dataflow edge pred -> consumer: FINN's 2, BLOCK_FIFO_DEPTH when both nodes are in the SAME down / up block (the inter-block edges stay at 2)."""
        sp, sc = _stage(pred, self.geom, self.extras), _stage(consumer, self.geom, self.extras)
        return BLOCK_FIFO_DEPTH if sp == sc and self._deep_stage(sc) else 2

    def _edge_fifo(self, key, label, width, depth: int = 2) -> None:
        lut = fifo_memory(int(width), depth)["lut"]
        self._add_const(key, lut=lut, item=dict(kind="edge_fifo", name=label, width_bits=int(width), depth=int(depth), lut=float(lut)))

    def _up_block_fifos(self, d: dict) -> None:
        """Both join FIFOs of an up block: ext end (`skip FIFO`) and main end (`FIFO main`). The main branch is the one holding the upsample node. Per-option constants like the regular skip FIFO."""
        stage = self.extras[d["join"]].geom.stage
        a, b = d["branch_a"], d["branch_b"]
        main = a if any(self.kinds.get(n) == "upsample" for n in a) else b
        ext = b if main is a else a
        if not main or not ext or f"{stage}.main_proj.0" not in self.geom:
            return
        w_in, cout = self.geom[f"{stage}.main_proj.0"].win, self.extras[d["join"]].geom.cout
        dw = any(self.kinds.get(n) == "frozen_dw" for n in main)                    # the bilinear decoder: the main branch holds the frozen depthwise conv
        for node, role, fn, is_skip in ((ext[0], "up_skip", up_skip_depth_words, True), (main[0], "up_main_dw" if dw else "up_main", up_main_depth_words, False)):
            if node not in self.options:
                continue
            for key, _ in self.options[node]:
                pe = (self.cost[key].get("thr_pe") or key[1]) if node in self.geom else key[1]          # a conv's own threshold PE sets its output stream
                self._fifo_item(key, f"{d['join']}.{'skip' if is_skip else 'main'}", pe * self._bits(key), fn(w_in, cout, pe, dw), stage, is_skip=is_skip, producer=node, consumer=d["join"], role_kind=role)
            self.skip_nodes[node] = dict(join=d["join"], n_fill_px=w_in, conv=None, kind=role)
            self.replaced_edges.add((node, d["join"]))

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
                        cost = fcm.dwc_cost(w1, w2)["total_lut"] + fifo_memory(int(w2), self._edge_depth(pred, consumer))["lut"]      # the DWC node and the FIFO behind it (depth 2; BLOCK_FIFO_DEPTH inside a down / up block)
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
        tot = dict(lut=0.0, bram18=0.0, uram18=0.0, dwc_lut=0.0, edge_fifo_lut=0.0, n_edge_fifos=0, n_deeper_edge_fifos=0)      # n_deeper: edge FIFOs deeper than 2 (up-block ext branch)
        for name, key in chosen.items():
            for it in self.items.get(key, []):
                if it["kind"] == "edge_fifo":
                    tot["edge_fifo_lut"] += it["lut"]
                    tot["n_edge_fifos"] += 1
                    tot["n_deeper_edge_fifos"] += it["depth"] > 2
                elif it["kind"] == "fifo":
                    fifos.append(self._bridge_entry(it, chosen))
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
            tot["edge_fifo_lut"] += fifo_memory(int(w2), self._edge_depth(pred, consumer))["lut"]
            tot["n_edge_fifos"] += 1
            tot["n_deeper_edge_fifos"] += self._edge_depth(pred, consumer) > 2
        return dict(fifos=fifos, dwcs=dwcs, totals=tot)

    def _bridge_entry(self, it: dict, chosen: dict) -> dict:
        """One entry of the flat `intra_block_fifos` list in the schema (and role vocabulary) net_fold.py writes and hardware/finn_s12_build_steps.py reads:
        producer / consumer = '<stage>.<analytical role>' (skip: thr_s => add; prefetch: dwc or thr_r => fmpad, initial block: dup => fmpad); the MILP names ride along as *_milp."""
        stage = it["stage"]
        if it["role_kind"] == "skip":
            producer, consumer = f"{stage}.thr_s", f"{stage}.add"
        elif it["role_kind"] == "up_skip":
            producer, consumer = f"{stage}.thr_e", f"{stage}.add"
        elif it["role_kind"] == "up_main":
            producer, consumer = f"{stage}.thr_s", f"{stage}.add"
        elif it["role_kind"] == "up_main_dw":                                              # bilinear decoder: the main branch ends in the depthwise conv's threshold (Thr_k)
            producer, consumer = f"{stage}.thr_k", f"{stage}.add"
        elif it["role_kind"] == "prefetch_dw":                                             # in front of the frozen depthwise conv's FMPadding (role fmpad_k), fed by the nearest upsampler (a DWC in between when widths differ)
            consumer = f"{stage}.fmpad_k"
            preds = [p for p in self.dmap.get(it["conv"], []) if p in chosen]
            in_w = self.ports(chosen[it["conv"]])["in_w"]
            producer = f"{stage}.dwc" if preds and self.ports(chosen[preds[0]])["out_w"] != in_w else f"{stage}.upnn"
        else:
            consumer = f"{stage}.fmpad"
            if stage == "initial":
                producer = "initial.dup"
            else:
                preds = [p for p in self.dmap.get(it["conv"], []) if p in chosen]
                in_w = self.ports(chosen[it["conv"]])["in_w"]
                producer = f"{stage}.dwc" if preds and self.ports(chosen[preds[0]])["out_w"] != in_w else f"{stage}.thr_r"
        mem = it["mem"]
        return dict(
            stage=stage, name=(f"{stage}.skip_FIFO" if it["is_skip"] else (f"{stage}.FIFO_main" if it["role_kind"] in ("up_main", "up_main_dw") else f"{stage}.prefetch")), producer=producer, consumer=consumer,
            producer_milp=it["producer_milp"], consumer_milp=it["consumer_milp"], depth=it["depth"], width_bits=it["width_bits"], max_occupancy=it["depth"],
            is_skip=it["is_skip"], mem=mem, finn_impl=("rtl" if mem == "srl" else f"vivado/{'ultra' if mem == 'uram' else 'block'}"),
            mem_bram18=it["mem_bram18"], mem_lut=it["lut"], depth_alloc=it["depth_alloc"], mem_uram18=it["mem_uram18"])

    def _chosen_width(self, chosen, name, side, pred=None):
        key = chosen[name]
        return self.in_w_for_edge(key, pred) if side == "in" else self.ports(key)["out_w"]

    def inter_block_fifos(self, chosen: dict, depth: int = 2) -> list[dict]:
        """One depth-`depth` FIFO per dataflow edge leaving a block (the same list net_fold.py writes). REPORT only: since 2026-10-11 these edges are priced inside the solve by
        _build_edge_fifos (together with all the other depth-2 edge FIFOs), so their LUT must not be added to the totals a second time."""
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
