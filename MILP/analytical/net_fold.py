"""Assemble a whole-network folding from the ANALYTICAL per-block models, written in the MILP flow's own formats.

    python3 MILP/analytical/net_fold.py --config config_12_dense_relu_nearest_upsample_256 --bits 6 --fps 250 --clock-mhz 100 \\
        --tag U4_int6_fps250 --out-dir MILP/artifacts/U4_256_int6_fps250_analytical_v1

Pipeline (the MILP's own plumbing is reused, nothing is re-derived):
  1. finn_milp.build_model_and_graph() traces the network (LayerQuantEnetFINN has the same layer / site names as ENet) -> LayerGeometry per conv / pool,
     ExtraNode per join / dup / threshold, and the dataflow graph.
  2. Blocks are visited in dataflow order. Each one runs its analytical model (model_int_bottleneck / model_dn_bottleneck / model_bottleneck /
     model_up_bottleneck / model_fnl_block) at the block's frame-cycle budget, which gives PE / SIMD of every MVAU, the PE of the thresholds, the
     joins (Add, residual / out_act merged, skip), the pad-MVAU of the downsampling skip, and the PE rule of the nearest-upsampler join chain.
  3. Those folds are written under the MILP layer / extra-node names with the MILP cost functions (layer_cost_pe_simd, extra_node_options), so
     per_layer / extra_nodes / _diagnostics / dataflow_graph have exactly the schema of finn_milp.py's output.
     expand_layer_bits.py then expands the bits to the per-quantizer-site file (layer_bits_SITES_<tag>.json).
  4. Standard inter-block FIFOs (top-level key "inter_block_fifos", ignored by consumers that do not know it): one per dataflow edge that leaves a block,
     sized to fill one BRAM18 at its stream width (UG573 SDP aspect table), forced to BRAM.

Rate policy:
  * every node's cycles per frame <= F = clock / fps (250 fps at 100 MHz = 400,000 cycles); the analytical search takes the largest cycle count that
    fits, so each stage is matched to the target;
  * optional latency cap --max-latency-ms (the MILP's measure: sum of every node's cycles per frame / clock): the per-node budget is bisected down until it holds;
  * "downstream faster than upstream": block k gets budget min(F, max(floor * F, R * slowest_{k-1})), R = --dsr-ratio (the MILP's own DSR ratio, default
    1.04), floor = --budget-floor. A node that cannot be slowed (the lattice of legal (PE, SIMD) pairs) keeps the chain from relaxing past the floor,
    and the achieved MILP DSR ratio (cycles per output element, downstream vs. upstream) is reported in _diagnostics.chain_rate_imbalance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "enet", REPO / "MILP", REPO / "MILP" / "utils", REPO / "MILP" / "analytical"):
    sys.path.insert(0, str(p))

import finn_milp  # noqa: E402
import finn_cost_model as fcm  # noqa: E402
from finn_cost_model import calibrated_bram18k, calibrated_lut, layer_cost_pe_simd_auto_ram  # noqa: E402
from finn_milp import (  # noqa: E402
    EXTRA_OP_LABEL, RAM_STYLE_AUTO, STREAM_NODE_KINDS, THRESHOLD_KINDS, VARIANT_RTL_DSP_NOACT1, XCZU7EV, _calibration_force_dsp, _variant_cost_kwargs,
    build_model_and_graph, extra_node_options,
)
from milp_outputs import compute_branch_imbalance_report, compute_chain_rate_imbalance_report  # noqa: E402

from bottleneck import _retarget_threshold, model_bottleneck  # noqa: E402
from dn_bottleneck import model_dn_bottleneck  # noqa: E402
from fnl_block import model_fnl_block  # noqa: E402
from int_bottleneck import model_int_bottleneck  # noqa: E402
from up_bottleneck import model_up_bottleneck  # noqa: E402

_XOPT_CACHE: dict = {}


def xopts(node):
    """finn_milp.extra_node_options(node, force_dsp=True), cached per node (it is called for every node on every budget trial)."""
    key = (node.geom.name, finn_milp.CANDIDATE_BITS)
    if key not in _XOPT_CACHE:
        _XOPT_CACHE[key] = extra_node_options(node, True)
    return _XOPT_CACHE[key]


THR_RAM_STYLE = "block"              # the analytical models price every standalone threshold with ram_style="block" (depth_trigger_bram 1024 in the probes)
_BRAM18_ASPECTS = ((1, 16384), (2, 8192), (4, 4096), (9, 2048), (18, 1024), (36, 512))   # UG573 table 1-10 (width, depth)
FOLDABLE = ("MVAU", "Thr", "Add", "Dup")


# ---------------------------------------------------------------------------------------------- block bookkeeping
def block_order(geoms) -> list[str]:
    seen: list[str] = []
    for g in geoms:
        if g.stage not in seen:
            seen.append(g.stage)
    return seen


def block_kind(stage: str) -> str:
    if stage == "initial":
        return "init"
    if stage == "final":
        return "final"
    if stage.startswith("down"):
        return "dn"
    if stage.startswith("up"):
        return "up"
    return "reg"


def run_block(stage: str, geom: dict, bits: int, F_k: int):
    """Analytical model of one block at budget F_k cycles per frame -> (BlockResult, layer folds, extra folds)."""
    kind = block_kind(stage)
    lf: dict[str, tuple] = {}     # MILP layer name -> (pe, simd, thr_pe or None, analytical frame cycles of the MVAU)
    xf: dict[str, tuple] = {}     # MILP extra node name -> (pe, simd, ram_style or None)

    def put(layer, mv, thr=None):
        lf[layer] = (mv.pe, mv.simd, thr.pe if thr is not None else None, mv.frame_cycles)

    if kind == "reg":
        r0 = geom[f"{stage}.reduce.0"]
        cin, cmid = r0.cin, r0.cout
        d = geom[f"{stage}.conv"].dh
        px = r0.hin * r0.win
        r = model_bottleneck(cin, cin // cmid, cin // cmid, max(1, F_k // px), bits, r0.hin, r0.win, k=3, dilation=d)
        n = {x.name: x for x in r.nodes}
        put(f"{stage}.reduce.0", n["MVAU_r"], n["Thr_r"]); put(f"{stage}.conv", n["MVAU_m"], n["Thr_m"]); put(f"{stage}.expand.0", n["MVAU_e"], n["Thr_e"])
        xf[f"{stage}.add"] = (n["Add"].pe, 1, None)
        xf[f"{stage}.skip_quant"] = (n["Thr_s"].pe, 1, THR_RAM_STYLE)
        xf[f"{stage}.residual_add"] = (n["Thr_out"].pe, 1, THR_RAM_STYLE)
        xf[f"{stage}.out_act"] = (n["Thr_out"].pe, 1, THR_RAM_STYLE)
    elif kind == "dn":
        pool = geom[f"{stage}.pool"]
        cin, cout = pool.cin, geom[f"{stage}.expand.0"].cout
        cmid = geom[f"{stage}.reduce.0"].cout
        r = model_dn_bottleneck(cin, cout, cout // cmid, bits, pool.hin, pool.win, T_out=F_k / (pool.hout * pool.wout), skip_order="pad_thr", skip_pad="mvau")
        n = {x.name: x for x in r.nodes}
        put(f"{stage}.reduce.0", n["MVAU_r"], n["Thr_r"]); put(f"{stage}.conv.0", n["MVAU_m"], n["Thr_m"]); put(f"{stage}.expand.0", n["MVAU_e"], n["Thr_e"])
        xf[f"{stage}.skip_pad"] = (n["MVAU_s"].pe, n["MVAU_s"].simd, "auto")
        xf[f"{stage}.skip_quant"] = (n["Thr_s"].pe, 1, THR_RAM_STYLE)
        xf[f"{stage}.add"] = (n["Add"].pe, 1, None)
        xf[f"{stage}.residual_add"] = (n["Thr_out"].pe, 1, THR_RAM_STYLE)
        xf[f"{stage}.out_act"] = (n["Thr_out"].pe, 1, THR_RAM_STYLE)
    elif kind == "up":
        mp = geom[f"{stage}.main_proj.0"]
        cin, cout, cmid = mp.cin, mp.cout, geom[f"{stage}.reduce.0"].cout
        r = model_up_bottleneck(cin, cout, cin // cmid, bits, mp.hin, mp.win, T_out=F_k / (4 * mp.hin * mp.win), skip_conv=False)
        n = {x.name: x for x in r.nodes}
        put(f"{stage}.main_proj.0", n["MVAU_p"], n["Thr_p"]); put(f"{stage}.reduce.0", n["MVAU_r"], n["Thr_r"])
        put(f"{stage}.up.0", n["MVAU_u"], n["Thr_u"]); put(f"{stage}.expand.0", n["MVAU_e"], n["Thr_e"])
        xf[f"{stage}.skip_quant"] = (n["Thr_s"].pe, 1, THR_RAM_STYLE)
        xf[f"{stage}.add"] = (n["Add"].pe, 1, None)
        xf[f"{stage}.residual_add"] = (n["Thr_out"].pe, 1, THR_RAM_STYLE)
        xf[f"{stage}.out_act"] = (n["Thr_out"].pe, 1, THR_RAM_STYLE)
    elif kind == "init":
        c = geom["initial.conv"]
        r = model_int_bottleneck(c.cin, c.cin + c.cout, bits, c.hin, c.win, F=F_k)
        n = {x.name: x for x in r.nodes}
        put("initial.conv", n["MVAU_c"], n["Thr_c"])
        xf["initial.input_quant"] = (n["Thr_in"].pe, 1, THR_RAM_STYLE)
        xf["initial.act"] = (n["Thr_act"].pe, 1, THR_RAM_STYLE)
        xf["initial.pool_quant"] = (n["Thr_m"].pe, 1, THR_RAM_STYLE)       # the branch-quant threshold UPSTREAM of the maxpool (MILP kind pool_quant)
    else:   # final (LayerQuantEnetFINN final_bias=True does not lower to hardware: the probes use the no-bias layer)
        f = geom["final"]
        r = model_fnl_block(f.cin, f.cout, bits, f.hin, f.win, F=F_k, bias=False)
        n = {x.name: x for x in r.nodes}
        lf["final"] = (n["MVAU_f"].pe, n["MVAU_f"].simd, None, n["MVAU_f"].frame_cycles)
    slowest = max(x.frame_cycles for x in r.nodes if x.name.startswith(FOLDABLE))
    return r, lf, xf, slowest


# ---------------------------------------------------------------------------------------------- MILP-format entries
def layer_entry(geom, bits, pe, simd, thr_pe, stage, wbits=None, no_thr=False):
    """per_layer entry priced EXACTLY like the analytical models price a layer: layer_cost_pe_simd_auto_ram(force_dsp=True), then the standalone threshold
    re-targeted to the analytical PE_t (bottleneck._retarget_threshold, ram_style block). no_thr: the layer has no threshold (the final logits layer)."""
    wb = wbits or bits
    if geom.op_type == "MaxPool2d":
        pe, simd = 1, 1
        cost = layer_cost_pe_simd_auto_ram(geom, wb, bits, 1, 1, force_dsp=True)
    else:
        cost = layer_cost_pe_simd_auto_ram(geom, wb, bits, pe, simd, force_dsp=True)
        if thr_pe is not None:
            cost = _retarget_threshold(cost, geom, bits, thr_pe)
    cost = {k: v for k, v in cost.items() if k != "ram_style_chosen"}
    if no_thr:
        cost = {**cost, "total_lut": cost["total_lut"] - cost["thr_lut"], "thr_lut": 0.0, "thr_bram18": 0.0, "thr_uram18": 0.0}
    vk = _variant_cost_kwargs(VARIANT_RTL_DSP_NOACT1, True)
    e = {"stage": stage, "pe": pe, "simd": simd, "thr_ram_style": THR_RAM_STYLE, "variant": VARIANT_RTL_DSP_NOACT1, "force_dsp": vk["force_dsp"],
         "mvau_noAct": vk["no_activation"], "weight_bits": wb, "act_bits": bits, **cost}
    e["lut_calibrated"] = calibrated_lut(e["total_lut"], wb, bits, force_dsp=_calibration_force_dsp(vk))
    e["bram18k_calibrated"] = calibrated_bram18k(e["swu_bram18"] + e["wm_bram18"] + e.get("thr_bram18", 0), wb, bits, force_dsp=True)
    return e


def _x_entry(node, pe, simd, ram_style, bits, cycles, lut, bram, dsp=0, uram=0.0, w=None, a=None, **extra):
    return {
        "kind": node.kind, "stage": node.geom.stage, "pe": pe, "simd": simd, "ram_style": ram_style, "weight_bits": w, "act_bits": a,
        "bit_sources": list(node.bit_sources), "channels": node.geom.cout, "cycles": cycles, "lut_calibrated": lut, "bram18k_calibrated": bram,
        "dsp": dsp, "uram18": uram, **extra,
    }


def pad_mvau_cost(node, bits, pe, simd):
    """The downsampling skip's 1x1 pad-MVAU as dn_bottleneck prices it: INT8 weights, network activation bits, its threshold (PE_t = Thr_s PE) belongs to the
    skip threshold node, not to the MVAU."""
    return layer_cost_pe_simd_auto_ram(node.geom, 8, bits, pe, simd, force_dsp=True)


def extra_entry(node, bits, pe, simd, ram_style, pad_cost=None):
    """Extra-node entry priced like the analytical models: threshold_node_cost (block RAM, default input width) for every standalone threshold, stream_node_cost
    for dup / add / concat / upsample. residual_add is MERGED into out_act (the compose pass; analytical Thr_out): kept in the graph at zero cost."""
    kind, g = node.kind, node.geom
    rk = _variant_cost_kwargs(VARIANT_RTL_DSP_NOACT1, True)
    if kind == "residual_add":
        return _x_entry(node, pe, simd, ram_style, bits, 0, 0.0, 0.0, a=bits, merged_into=g.name[: -len("residual_add")] + "out_act")
    if kind == "pad_mvau":
        c = pad_cost
        return _x_entry(node, pe, simd, "auto", bits, c["mvu_cycles"], calibrated_lut(c["total_lut"] - c["thr_lut"], 8, bits, force_dsp=_calibration_force_dsp(rk)),
                        calibrated_bram18k(c["swu_bram18"] + c["wm_bram18"], 8, bits, force_dsp=rk["force_dsp"]), dsp=c["total_dsp"],
                        uram=c["wm_uram18"] + c.get("swu_uram18", 0), w=8, a=bits, threshold_in="skip_quant")
    if kind == "skip_quant" and pad_cost is not None:         # downsampling skip: the threshold retargeted in the pad-MVAU's own cost (in_bits = its accumulator)
        lut, bram, uram = fcm._thresholding_rtl_cost(pe, bits, g.cout, ram_style="block", in_bits=pad_cost["acc_bits"])
        return _x_entry(node, pe, 1, "block", bits, g.hout * g.wout * math.ceil(g.cout / pe), calibrated_lut(lut, bits, bits, force_dsp=_calibration_force_dsp(rk)),
                        calibrated_bram18k(bram, bits, bits, force_dsp=rk["force_dsp"]), uram=uram, a=bits)
    if kind in THRESHOLD_KINDS:
        c = fcm.threshold_node_cost(g, bits, pe)
        return _x_entry(node, pe, 1, "block", bits, c["cycles"], calibrated_lut(c["total_lut"], bits, bits, force_dsp=_calibration_force_dsp(rk)),
                        calibrated_bram18k(c["thr_bram18"], bits, bits, force_dsp=rk["force_dsp"]), uram=c["thr_uram18"], a=bits)
    c = fcm.stream_node_cost(kind, g, pe) if kind in fcm.FOLDABLE_STREAM_KINDS else fcm.stream_node_cost(kind, g)
    return _x_entry(node, pe if kind in fcm.FOLDABLE_STREAM_KINDS else 1, 1, "none", bits, c["cycles"], c["total_lut"], c["swu_bram18"] + c["wm_bram18"] + c["thr_bram18"])


def best_extra_fold(node, bits, budget):
    """Fallback (dup, concat, upsample, anything not mapped from an analytical node): the legal option with the largest cycle count <= budget, then cheapest."""
    opts = [(k, c) for k, c in xopts(node) if node.kind not in THRESHOLD_KINDS or k[6] == bits]
    fit = [(k, c) for k, c in opts if c["cycles"] <= budget] or [min(opts, key=lambda kc: kc[1]["cycles"])]
    k, c = max(fit, key=lambda kc: (kc[1]["cycles"], -kc[1]["total_lut"]))
    return k[1], k[2], k[3]


def bram_fifo(width_bits: int) -> dict:
    """One-BRAM18 standard FIFO: the narrowest SDP aspect that holds the stream width gives the depth; wider streams span several BRAM18 at depth 512."""
    for w, d in _BRAM18_ASPECTS:
        if w >= width_bits:
            return {"depth": d, "bram18": 1, "aspect": [w, d]}
    w, d = _BRAM18_ASPECTS[-1]
    return {"depth": d, "bram18": math.ceil(width_bits / w), "aspect": [w, d]}


def write_sites(config: str, folding_json: Path, sites_json: Path) -> None:
    """layer_bits_SITES_<tag>.json = MILP/expand_layer_bits.py run in-process. The working-tree LayerQuantENet.py lacks its `from QuantENet import (...)`
    block, VALID_CONTEXT_PATTERNS and ACT_SITE_TYPES (dropped by the 'Sync' commit 66e0590e8b, present at b16b35468a; the file is not touched here), so what it needs from QuantENet (helpers, VALID_CONTEXT_PATTERNS, the dilation patterns) is injected first, as the probe export scripts do."""
    import nnunetv2.nets.LayerQuantENet as lq
    import nnunetv2.nets.CombinedQuantENet as cq
    import nnunetv2.nets.ENet as en
    import nnunetv2.nets.QuantENet as qe
    import brevitas.nn as qnn
    for mod in (qe, cq, en):      # QuantENet helpers, CombinedQuantENet.VALID_CONTEXT_PATTERNS, ENet's dilation patterns
        for name in dir(mod):
            if not name.startswith("__") and not hasattr(lq, name):
                setattr(lq, name, getattr(mod, name))
    if not hasattr(lq, "ACT_SITE_TYPES"):      # as defined in LayerQuantENet.py at b16b35468a (the later "Sync" commit dropped it)
        lq.ACT_SITE_TYPES = (qnn.QuantReLU, qnn.QuantIdentity, qe.QuantDecomposedLeakyAct, qe.QuantFusedLeakyAct, qnn.QuantEltwiseAdd)
    import expand_layer_bits
    old_argv = sys.argv
    sys.argv = ["expand_layer_bits.py", "--config", config, "--ilp-result", str(folding_json), "--out-file", str(sites_json)]
    try:
        expand_layer_bits.main()
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------------------------- assembly
def assemble(a, ctx, F_top: int):
    """One whole-network assembly with per-node budget F_top (cycles per frame): the analytical models per block (optionally ratcheted so that every
    block is no slower than the one before it), costed with the MILP functions. Returns the MILP-layout result dict."""
    geoms, extras, geom, xnode, dmap, kinds = ctx
    per_layer: dict[str, dict] = {}
    extra_out: dict[str, dict] = {}
    profile = []
    slowest_prev = None
    mismatches = []
    rough_latency = 0
    for stage in block_order(geoms):
        kind = block_kind(stage)
        F_k = F_top if (a.no_ratchet or slowest_prev is None) else int(min(F_top, max(a.budget_floor * F_top, a.dsr_ratio * slowest_prev)))
        r, lf, xf, slowest = run_block(stage, geom, a.bits, F_k)
        rough_latency += r.latency_first_out_cycles
        for lname, (pe, simd, thr_pe, an_cycles) in lf.items():
            e = layer_entry(geom[lname], a.bits, pe, simd, thr_pe, stage, no_thr=(lname == "final"))
            if e["mvu_cycles"] != an_cycles:
                mismatches.append((lname, an_cycles, e["mvu_cycles"]))
            per_layer[lname] = e
        for lg in (g for g in geoms if g.stage == stage and g.name not in per_layer):      # layers the analytical block has no node for (maxpool)
            per_layer[lg.name] = layer_entry(lg, a.bits, 1, 1, None, stage)
        pad_cost = None
        if f"{stage}.skip_pad" in xf:                    # downsampling skip: the pad-MVAU's cost also carries the skip threshold's accumulator width
            ppe, psimd, _ = xf[f"{stage}.skip_pad"]
            pad_cost = pad_mvau_cost(xnode[f"{stage}.skip_pad"], a.bits, ppe, psimd)
        for xn in (n for n in extras if n.geom.stage == stage):
            pe, simd, rs = xf.get(xn.geom.name) or best_extra_fold(xn, a.bits, F_k)
            extra_out[xn.geom.name] = extra_entry(xn, a.bits, pe, simd, rs, pad_cost=pad_cost if xn.kind in ("pad_mvau", "skip_quant") else None)
            extra_out[xn.geom.name]["bits_rule"] = "fixed" if xn.fixed_bits is not None else ("max_of_sources" if xn.bit_sources else "n/a")
        profile.append(dict(block=stage, kind=kind, budget=F_k, slowest_foldable=slowest, ratio_to_prev=(round(slowest / slowest_prev, 3) if slowest_prev else None)))
        slowest_prev = slowest
    for xn in extras:       # fork-source dups named after a previous block's out_act can belong to no block above
        if xn.geom.name not in extra_out:
            pe, simd, rs = best_extra_fold(xn, a.bits, F_top)
            extra_out[xn.geom.name] = extra_entry(xn, a.bits, pe, simd, rs)
            extra_out[xn.geom.name]["bits_rule"] = "fixed" if xn.fixed_bits is not None else ("max_of_sources" if xn.bit_sources else "n/a")

    # ---- inter-block FIFOs: one per dataflow edge that leaves a block
    def stage_of(n):
        return per_layer[n]["stage"] if n in per_layer else extra_out[n]["stage"]

    def width_of(n):
        if n in extra_out:
            return extra_out[n]["pe"] * a.bits
        return (geom[n].cout if geom[n].op_type == "MaxPool2d" else per_layer[n].get("thr_pe", 1)) * a.bits
    known = per_layer.keys() | extra_out.keys()
    fifos = []
    for name, preds in dmap.items():
        for pr in preds:
            if pr in known and name in known and stage_of(pr) != stage_of(name):
                w = width_of(pr)
                bf = bram_fifo(w)
                fifos.append({"name": f"{pr}->{name}", "producer": pr, "consumer": name, "width_bits": w, "mem": "bram", **bf,
                              "efficiency": round(w * bf["depth"] / (bf["bram18"] * 18432), 3)})

    # ---- totals / diagnostics in the MILP layout
    L, X = list(per_layer.values()), list(extra_out.values())
    total_lut = sum(v["lut_calibrated"] for v in L + X)
    total_bram = sum(v["bram18k_calibrated"] for v in L + X)
    total_uram = sum(v.get("wm_uram18", 0) + v.get("swu_uram18", 0) + v.get("thr_uram18", 0) for v in L) + sum(v["uram18"] for v in X)
    total_dsp = sum(v["total_dsp"] for v in L) + sum(v["dsp"] for v in X)
    total_cycles = sum(v["cycles"] for v in L + X)
    fifo_bram = sum(f["bram18"] for f in fifos)
    node_cycles = {**{n: v["cycles"] for n, v in per_layer.items()}, **{n: v["cycles"] for n, v in extra_out.items()}}
    bott = max(node_cycles, key=node_cycles.get)
    hardware_nodes = [*geoms, *(n.geom for n in extras)]
    result = {
        "status": "Optimal",
        "layer_weight_bits": {n: a.bits for n in per_layer},
        "layer_act_bits": {n: a.bits for n in per_layer},
        "per_layer": per_layer,
        "extra_nodes": extra_out,
        "inter_block_fifos": fifos,
        "_diagnostics": {
            "n_layers": len(per_layer), "n_extra_nodes": len(extra_out), "force_dsp": True, "min_resources": True,
            "total_lut_calibrated": total_lut, "xczu7ev_lut_budget": XCZU7EV["LUT"], "lut_pct_of_budget": 100 * total_lut / XCZU7EV["LUT"],
            "total_bram18k_calibrated": total_bram, "xczu7ev_bram18k_budget": XCZU7EV["BRAM_18K"], "bram_pct_of_budget": 100 * total_bram / XCZU7EV["BRAM_18K"],
            "inter_block_fifo_bram18": fifo_bram, "total_bram18k_with_inter_block_fifos": total_bram + fifo_bram,
            "total_uram18": total_uram, "xczu7ev_uram_budget": XCZU7EV["URAM"], "uram_pct_of_budget": 100 * total_uram / XCZU7EV["URAM"],
            "total_cycles": total_cycles, "latency_ms": total_cycles / (a.clock_mhz * 1e3),
            "total_dsp": total_dsp, "xczu7ev_dsp_budget": XCZU7EV["DSP"], "dsp_pct_of_budget": 100 * total_dsp / XCZU7EV["DSP"],
            "bottleneck_node": bott, "bottleneck_cycles": node_cycles[bott],
            "max_node_cycles": int(a.clock_mhz * 1e6 / a.fps), "per_node_budget_used": F_top, "target_fps": a.fps, "clock_mhz": a.clock_mhz,
            "max_latency_ms": a.max_latency_ms, "fps": a.clock_mhz * 1e6 / node_cycles[bott],
            "analytical": {"source": "MILP/analytical/net_fold.py", "dsr_ratio": a.dsr_ratio, "budget_floor": a.budget_floor, "ratchet": not a.no_ratchet,
                           "sum_of_block_first_out_latencies_cycles": rough_latency, "block_profile": profile,
                           "mvau_cycle_mismatches_vs_milp_cost_model": mismatches},
            "note": "Folding assembled from the analytical per-block models (U4 widths), costed with the MILP's own layer_cost_pe_simd / extra_node_options. Not an ILP solve. "
                    "latency_ms is the MILP's measure (sum of every node's cycles per frame / clock); sum_of_block_first_out_latencies_cycles is the pipeline-fill estimate.",
        },
        "dataflow_graph": {
            "input": [finn_milp.IN_CHANNELS, *finn_milp.INPUT_HW], "edges": dmap,
            "shapes": {g.name: [g.cout, g.hout, g.wout] for g in hardware_nodes},
            "op_types": {**{g.name: g.op_type for g in geoms}, **{n.geom.name: EXTRA_OP_LABEL[n.kind] for n in extras}},
            "kinds": kinds,
        },
    }
    fixed = sorted([n for n, g in geom.items() if g.op_type == "MaxPool2d"] + [n for n, x in xnode.items() if x.kind in ("concat", "upsample")])
    result["_diagnostics"]["fixed_cycle_nodes"] = fixed
    return result


def finalize_reports(result):
    fixed = result["_diagnostics"]["fixed_cycle_nodes"]
    result["_diagnostics"]["branch_imbalance"] = compute_branch_imbalance_report(result)
    result["_diagnostics"]["chain_rate_imbalance"] = compute_chain_rate_imbalance_report(result, fixed)


# ---------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config_12_dense_relu_nearest_upsample_256")
    ap.add_argument("--bits", type=int, default=6, help="uniform weight = activation bits")
    ap.add_argument("--fps", type=float, default=250.0, help="throughput target: every node's cycles per frame <= clock / fps")
    ap.add_argument("--clock-mhz", type=float, default=100.0)
    ap.add_argument("--max-latency-ms", type=float, default=None,
                    help="MILP latency cap (sum of all node cycles per frame / clock): the per-node budget is lowered (bisection) until it holds")
    ap.add_argument("--dsr-ratio", type=float, default=1.04, help="tolerance of the downstream-faster ratchet (MILP --dsr-ratio default of the S12 runs)")
    ap.add_argument("--budget-floor", type=float, default=0.6, help="the ratchet never tightens a block below this fraction of the per-node budget")
    ap.add_argument("--no-ratchet", action="store_true", help="every block gets the full per-node budget (stages matched to it, no downstream-faster rule)")
    ap.add_argument("--tag", default="final", help="file tag: layer_bits_folding_<tag>.json / layer_bits_SITES_<tag>.json (S12 artifacts use 'final')")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--no-sites", action="store_true", help="skip the expand_layer_bits.py step")
    a = ap.parse_args()

    finn_milp.load_config(a.config)
    finn_milp.CANDIDATE_BITS = tuple(sorted(set(finn_milp.CANDIDATE_BITS) | {a.bits}))
    model, geoms, extras, _pred, dmap, kinds = build_model_and_graph()
    ctx = (geoms, extras, {g.name: g for g in geoms}, {n.geom.name: n for n in extras}, dmap, kinds)
    F = int(a.clock_mhz * 1e6 / a.fps)
    cap = a.max_latency_ms * a.clock_mhz * 1e3 if a.max_latency_ms else None
    print(f"throughput target {a.fps:g} fps @ {a.clock_mhz:g} MHz -> every node <= {F} cycles/frame"
          + (f"; latency cap {a.max_latency_ms:g} ms -> sum of node cycles <= {cap:.0f}" if cap else "")
          + f"; INT{a.bits} uniform; {len(geoms)} layers, {len(extras)} extra nodes", flush=True)

    result = assemble(a, ctx, F)
    if cap is not None and result["_diagnostics"]["total_cycles"] > cap:
        lo, hi = 81_920, F       # hi violates the cap; lo = the init maxpool floor (not foldable). Find the largest per-node budget in [lo, hi) that holds
        while True:
            try:
                best = assemble(a, ctx, lo)
                break
            except ValueError:   # some block's own floor (maxpool, UpsampleNearestNeighbour, FMPadding_Pixel) is above lo
                lo = int(lo * 1.05)
        if best["_diagnostics"]["total_cycles"] > cap:
            raise SystemExit(f"latency cap {a.max_latency_ms} ms infeasible: even at a per-node budget of {lo} cycles the sum is "
                             f"{best['_diagnostics']['latency_ms']:.1f} ms")
        while hi - lo > max(500, lo // 50):
            mid = (lo + hi) // 2
            trial = assemble(a, ctx, mid)
            ok = trial["_diagnostics"]["total_cycles"] <= cap
            print(f"  per-node budget {mid:7d}: sum {trial['_diagnostics']['latency_ms']:6.1f} ms  {'ok' if ok else 'over'}", flush=True)
            if ok:
                lo, best = mid, trial
            else:
                hi = mid
        result = best
    finalize_reports(result)
    d = result["_diagnostics"]

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"layer_bits_folding_{a.tag}.json"
    out.write_text(json.dumps(result, indent=2))
    prof = d["analytical"]["block_profile"]
    with open(a.out_dir / f"block_profile_{a.tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prof[0]))
        w.writeheader()
        w.writerows(prof)
    (a.out_dir / "run_args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2))
    if not a.no_sites:
        write_sites(a.config, out, a.out_dir / f"layer_bits_SITES_{a.tag}.json")
    cri = d["chain_rate_imbalance"]
    for pr in prof:
        print(f"  {pr['block']:12s} {pr['kind']:5s} budget {pr['budget']:7d}  slowest foldable {pr['slowest_foldable']:7d}")
    print(f"\nper-node budget used {d['per_node_budget_used']}  LUT {d['total_lut_calibrated']:.0f} ({d['lut_pct_of_budget']:.1f}%)  "
          f"BRAM18 {d['total_bram18k_calibrated']:.0f} + {d['inter_block_fifo_bram18']} inter-block FIFO  DSP {d['total_dsp']:.0f} ({d['dsp_pct_of_budget']:.1f}%)")
    print(f"bottleneck {d['bottleneck_node']} {d['bottleneck_cycles']} cyc -> {d['fps']:.1f} fps @ {a.clock_mhz:g} MHz; latency (sum of node cycles) {d['latency_ms']:.1f} ms; "
          f"pipeline-fill estimate {d['analytical']['sum_of_block_first_out_latencies_cycles'] / (a.clock_mhz * 1e3):.2f} ms")
    print(f"DSR (cycles per output element): max {cri.get('max_ratio')} median {cri.get('median_ratio')};  MVAU cycle mismatches vs MILP cost model: "
          f"{len(d['analytical']['mvau_cycle_mismatches_vs_milp_cost_model'])}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
