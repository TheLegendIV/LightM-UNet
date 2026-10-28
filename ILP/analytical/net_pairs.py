"""Block-pair stage of net_explicit.py: relax the INTER-block FIFOs and, when that is not enough, the PRODUCER's output-stage foldings of every adjacent block pair that runs more than 5 % slower than its two
blocks do alone. Resource constraints come from the MILP (the folding json's `_diagnostics.max_*_fraction`), and every changed node is repriced with the MILP's own cost model (finn_milp).

Why pairs: the whole-net chain showed the 22 % slowdown of S12_512_bilin_int8_maxfps lives in ONE boundary (regular5.0 -> final: a PE-1 8-bit output stream into a 32-bit-wide, zero-slack consumer); a pair run finds it in
minutes, a whole-net run takes ~13 minutes. Per pair (source of the producer PACED at the MILP target, 3 frames):
  * reference period = max(MILP target frame cycles, own period of the producer, own period of the consumer); mismatch = pair period / reference - 1;
  * mismatch at interface depth 2 <= tol  -> nothing to do;
  * else the interface FIFO is relaxed first (x4 per step, at most MAX_DOUBLINGS steps, kept only if the pair gets more than MIN_IMPROVEMENT faster);
  * if even a very deep interface does not bring the pair within tol, the PRODUCER's output chain (the expand MVAU, its threshold, the skip threshold, add, out_act) is widened one divisor of Cout at a time:
    repriced with finn_milp's cost functions, rejected when the design would break a cap, the producer block re-simulated (intra-block sizing) and the pair re-run.
Consumer-side widening is not done (not needed so far)."""
from __future__ import annotations

import copy
import math
import time
from multiprocessing import Pool

import finn_milp
import net_fifo
from bottleneck import MAX_DOUBLINGS, MIN_IMPROVEMENT, fifo_memory

IFACE_GROWTH = 4                       # an interface FIFO grows x4 per step
DEEP = 1 << 20                         # "unbounded" interface: tells bandwidth limits from depth limits and gives the peak occupancy
WIDEN_KINDS = ("reg", "dn", "up")      # producer block kinds whose output chain is widened (initial / final have no standard expand chain)

_BLOCKS: list | None = None            # set before a Pool is created (fork): the frozen block simulations
_CFG: dict = {}


def input_pixels(r) -> int:
    return int(r.params["height"] * r.params["width"])


def own_period(r) -> int:
    """The block's own simulated frame period (annotate_rate's value when it ran, else steady cyc/px x output pixels)."""
    v = r.verification
    return int(v["rate"]["period_cycles"]) if "rate" in v else int(v["steady_cyc_px"] * r.params["hout"] * r.params["wout"])


def reference_period(blocks: list, i: int, F: int) -> int:
    """What the pair i -> i+1 must not exceed by more than tol: the MILP target (a paced source cannot go faster) or the slower block's own period."""
    return max(F, own_period(blocks[i][2]), own_period(blocks[i + 1][2]))


def _pair_run(args):
    i, depth = args
    blocks, F = _BLOCKS, _CFG["F"]
    interval = F / input_pixels(blocks[i][2])
    caps = [net_fifo.capture_block(blocks[i][1], blocks[i][2], net_fifo.DEFAULT_FRAMES, interval),
            net_fifo.capture_block(blocks[i + 1][1], blocks[i + 1][2], net_fifo.DEFAULT_FRAMES, 0)]
    res = net_fifo.run_chain(net_fifo.compose(caps, [depth]), max_cycles=int(net_fifo.DEFAULT_FRAMES * F * 3) + 400000)
    period = None if res["period"] == float("inf") else int(res["period"])
    return dict(i=i, depth=depth, deadlock=bool(res["deadlock"]), period=period, peak=int(res["iface_max"][0]))


def _init_pairs(blocks, F):
    global _BLOCKS
    _BLOCKS, _CFG["F"] = blocks, F


def run_pairs(blocks: list, tasks: list[tuple[int, int]], F: int, workers: int) -> dict:
    """{(pair, depth): result} -- every task is one paced pair run on the (frozen) blocks."""
    _init_pairs(blocks, F)
    if workers > 1 and len(tasks) > 1:
        with Pool(min(workers, len(tasks)), initializer=_init_pairs, initargs=(blocks, F)) as pool:      # initializer: spawn (Windows) workers do not inherit the globals
            out = pool.map(_pair_run, tasks)
    else:
        out = [_pair_run(t) for t in tasks]
    return {(r["i"], r["depth"]): r for r in out}


# ------------------------------------------------------------------------------------------------ resources (the MILP's model)
def caps_and_totals(d: dict) -> tuple[dict, dict]:
    """Caps (absolute) and the design's current totals, both from the folding json's _diagnostics (the MILP's own numbers: nodes + modelled FIFOs / DWCs)."""
    g = d["_diagnostics"]
    dev = finn_milp.XCZU7EV
    caps = {"lut": g["max_lut_fraction"] * dev["LUT"], "bram": g["max_bram_fraction"] * dev["BRAM_18K"], "dsp": g["max_dsp_fraction"] * dev["DSP"], "uram": g["max_uram_fraction"] * dev["URAM"]}
    tot = {"lut": g["total_lut_calibrated"], "bram": g["total_bram18k_calibrated"], "dsp": g["total_dsp"], "uram": g.get("total_uram18", 0.0)}
    return caps, tot


def interface_delta(d: dict, stage: str, depth: int) -> dict:
    """Extra LUT / BRAM18 / URAM of the inter-block FIFO leaving `stage` at `depth` against depth 2 (the MILP's own inter_block_fifos entry gives its width; net_explicit's fifo_memory prices it)."""
    e = next((f for f in d.get("inter_block_fifos", []) if f["src_block"] == stage), None)
    if e is None:
        return {}
    m1, m0 = fifo_memory(e["width_bits"], depth), fifo_memory(e["width_bits"], net_fifo.MIN_DEPTH)
    return {"lut": m1["lut"] - m0["lut"], "bram": m1["bram18"] - m0["bram18"], "uram": m1["uram"] - m0["uram"]}


def widen_producer(d: dict, stage: str, kind: str, ctx, force_dsp: bool = True):
    """One divisor step up of the producer block's OUTPUT chain: `<stage>.expand.0` (PE, and its threshold: thr_pe follows the cost model) and the extra nodes skip_quant, add, out_act, residual_add.
    Every node is repriced with finn_milp's cost functions exactly as the solver does. Returns (new folding json, change record) or (None, reason)."""
    if kind not in WIDEN_KINDS:
        return None, f"block kind {kind!r} has no standard output chain to widen"
    geoms, extras, geom, xnode, dmap, kinds = ctx
    name = f"{stage}.expand.0"
    L = d["per_layer"][name]
    g = geom[name]
    steps = [x for x in finn_milp.divisors(g.cout) if x > L["pe"]]
    if not steps:
        return None, f"{name}: PE {L['pe']} is already the largest divisor of Cout = {g.cout}"
    pe1 = min(steps)
    d2 = copy.deepcopy(d)
    changed, delta = [], {"lut": 0.0, "bram": 0.0, "dsp": 0.0, "uram": 0.0}

    vk = finn_milp._variant_cost_kwargs(L["variant"], force_dsp)
    cost = finn_milp.layer_cost_pe_simd(g, L["weight_bits"], L["act_bits"], pe1, L["simd"], finn_milp.RAM_STYLE_AUTO, swu_ram_style=finn_milp._variant_swu_style(L["variant"]),
                                        thr_ram_style=L["thr_ram_style"], **vk)
    new = dict(L)
    new.update(cost)
    new["pe"] = pe1
    new["lut_calibrated"] = finn_milp.calibrated_lut(cost["total_lut"], L["weight_bits"], L["act_bits"], force_dsp=finn_milp._calibration_force_dsp(vk))
    new["bram18k_calibrated"] = finn_milp.calibrated_bram18k(cost["swu_bram18"] + cost["wm_bram18"] + cost.get("thr_bram18", 0), L["weight_bits"], L["act_bits"], force_dsp=vk["force_dsp"])
    old_c, new_c = {"lut": L["lut_calibrated"], "bram": L["bram18k_calibrated"], "dsp": L["total_dsp"], "uram": L.get("wm_uram18", 0) + L.get("swu_uram18", 0) + L.get("thr_uram18", 0)}, \
        {"lut": new["lut_calibrated"], "bram": new["bram18k_calibrated"], "dsp": new["total_dsp"], "uram": new.get("wm_uram18", 0) + new.get("swu_uram18", 0) + new.get("thr_uram18", 0)}
    for k in delta:
        delta[k] += new_c[k] - old_c[k]
    d2["per_layer"][name] = new
    changed.append(dict(node=name, pe=[L["pe"], pe1], cycles=[L["cycles"], new["cycles"]], thr_pe=[L.get("thr_pe"), new.get("thr_pe")]))

    rtl_kwargs = finn_milp._variant_cost_kwargs(finn_milp.VARIANT_RTL_DSP_NOACT1, force_dsp)
    for k in ("skip_quant", "add", "out_act", "residual_add"):
        xn = f"{stage}.{k}"
        if xn not in d["extra_nodes"]:
            continue
        E = d["extra_nodes"][xn]
        node = xnode[xn]
        opt = [(key, c) for key, c in finn_milp.extra_node_options(node, force_dsp)
               if key[1] == pe1 and key[3] == E["ram_style"] and key[5] == (E["weight_bits"] or 0) and key[6] == (E["act_bits"] or 0)]
        if not opt:
            return None, f"{xn}: no cost option at PE {pe1} (ram {E['ram_style']})"
        key, c = opt[0]
        w, bits = key[5], key[6]
        if node.kind in finn_milp.STREAM_NODE_KINDS:
            lut, bram = c["total_lut"], c["swu_bram18"] + c["wm_bram18"] + c["thr_bram18"]
        else:
            lut = finn_milp.calibrated_lut(c["total_lut"], w or bits, bits, force_dsp=finn_milp._calibration_force_dsp(rtl_kwargs))
            bram = finn_milp.calibrated_bram18k(c["swu_bram18"] + c["wm_bram18"] + c["thr_bram18"], w or bits, bits, force_dsp=rtl_kwargs["force_dsp"])
        uram = c["wm_uram18"] + c.get("swu_uram18", 0) + c["thr_uram18"]
        delta["lut"] += lut - E["lut_calibrated"]
        delta["bram"] += bram - E["bram18k_calibrated"]
        delta["dsp"] += c["total_dsp"] - E["dsp"]
        delta["uram"] += uram - E["uram18"]
        d2["extra_nodes"][xn].update(pe=pe1, cycles=c["cycles"], lut_calibrated=lut, bram18k_calibrated=bram, dsp=c["total_dsp"], uram18=uram)
        changed.append(dict(node=xn, pe=[E["pe"], pe1], cycles=[E["cycles"], c["cycles"]]))
    g2 = d2["_diagnostics"]
    g2["total_lut_calibrated"] += delta["lut"]
    g2["total_bram18k_calibrated"] += delta["bram"]
    g2["total_dsp"] += delta["dsp"]
    g2["total_uram18"] = g2.get("total_uram18", 0.0) + delta["uram"]
    dev = finn_milp.XCZU7EV
    g2["lut_pct_of_budget"] = 100 * g2["total_lut_calibrated"] / dev["LUT"]
    g2["bram_pct_of_budget"] = 100 * g2["total_bram18k_calibrated"] / dev["BRAM_18K"]
    g2["dsp_pct_of_budget"] = 100 * g2["total_dsp"] / dev["DSP"]
    return d2, dict(stage=stage, pe_step=f"PE {L['pe']} -> {pe1}", nodes=changed, delta={k: round(v, 1) for k, v in delta.items()})


def check_caps(d: dict, extra_iface: dict | None = None) -> list[str]:
    """Names of the caps the design (its current _diagnostics totals + extra interface cost) would break."""
    caps, tot = caps_and_totals(d)
    ex = extra_iface or {}
    return [k for k in ("lut", "bram", "dsp", "uram") if tot[k] + ex.get(k, 0.0) > caps[k] * (1 + 1e-9)]


# ------------------------------------------------------------------------------------------------ the stage
def relax_pairs(blocks: list, d: dict, ctx, F: int, tol: float, max_iterations: int, workers: int, rebuild, resim, log=print):
    """blocks: the intra-block-sized blocks (NOT modified: changed blocks are replaced by new objects). rebuild(d_new) -> {stage: freshly built block (stage, kind, r)}; resim(blocks_subset) -> the same blocks
    simulated (intra-block sizing, rate annotated).
    Returns (blocks_new, d_new, depths, report, changes)."""
    n = len(blocks)
    blocks = list(blocks)
    depths = [net_fifo.MIN_DEPTH] * (n - 1)
    report, changes = [], []
    t0 = time.time()
    scan = run_pairs(blocks, [(i, 2) for i in range(n - 1)] + [(i, DEEP) for i in range(n - 1)], F, workers)
    log(f"  pair scan: {n - 1} pairs x (depth 2, unbounded) in {time.time() - t0:.0f} s", flush=True)
    d_cur = d
    for i in range(n - 1):
        sa, sb = blocks[i][0], blocks[i + 1][0]
        ref = reference_period(blocks, i, F)
        p2, pd = scan[(i, 2)], scan[(i, DEEP)]
        mis = lambda r: None if r["deadlock"] or r["period"] is None else r["period"] / ref - 1
        rec = dict(pair=f"{sa} -> {sb}", index=i, reference=ref, depth2_period=p2["period"], deep_period=pd["period"], depth2_mismatch_pct=None if mis(p2) is None else round(100 * mis(p2), 2),
                   deep_mismatch_pct=None if mis(pd) is None else round(100 * mis(pd), 2), peak_occupancy=pd["peak"], action="none", steps=[])
        if mis(p2) is not None and mis(p2) <= tol:
            report.append(rec)
            continue
        log(f"  FLAGGED {sa} -> {sb}: depth-2 period {p2['period']} ({rec['depth2_mismatch_pct']}% over {ref}), unbounded {pd['period']} ({rec['deep_mismatch_pct']}%), peak {pd['peak']} words", flush=True)
        # 1. a producer-folding step when even an unbounded interface does not bring the pair within tolerance
        step = 0
        while (mis(pd) is None or mis(pd) > tol) and step < max_iterations:
            step += 1
            kind = blocks[i][1]
            d_new, info = widen_producer(d_cur, sa, kind, ctx)
            if d_new is None:
                rec["steps"].append(dict(step=step, result="stopped", reason=info))
                log(f"    producer step {step}: {info}", flush=True)
                break
            over = check_caps(d_new)
            if over:
                rec["steps"].append(dict(step=step, result="rejected", reason=f"would break the cap(s): {over}", **info))
                log(f"    producer step {step} ({info['pe_step']}) REJECTED: would break {over}", flush=True)
                break
            fresh = rebuild(d_new)[sa]
            sim = resim([fresh])
            blocks[i] = sim[0]
            d_cur = d_new
            scan2 = run_pairs(blocks, [(i, 2), (i, DEEP)], F, workers)
            p2, pd = scan2[(i, 2)], scan2[(i, DEEP)]
            ref = reference_period(blocks, i, F)
            rec["steps"].append(dict(step=step, result="applied", depth2_period=p2["period"], deep_period=pd["period"], deep_mismatch_pct=None if mis(pd) is None else round(100 * mis(pd), 2), **info))
            changes.append(dict(info, pair=rec["pair"], reason=f"deep-interface pair period {scan[(i, DEEP)]['period']} vs reference {ref}"))
            log(f"    producer step {step} ({info['pe_step']}, +{info['delta']['lut']} LUT, +{info['delta']['dsp']} DSP): depth-2 {p2['period']}, unbounded {pd['period']}", flush=True)
        # 2. the interface FIFO: the smallest depth within tolerance (x4 steps), kept only when it makes the pair > MIN_IMPROVEMENT faster
        if mis(p2) is not None and mis(p2) <= tol:
            rec["action"] = "folding only" if step else "none"
        elif mis(pd) is not None and mis(pd) <= tol:
            cand = [net_fifo.MIN_DEPTH * IFACE_GROWTH ** k for k in range(1, MAX_DOUBLINGS + 1)]
            res = run_pairs(blocks, [(i, c) for c in cand], F, workers)
            ok = [c for c in cand if not res[(i, c)]["deadlock"] and res[(i, c)]["period"] is not None and res[(i, c)]["period"] / ref - 1 <= tol]
            best = min(ok) if ok else None
            gate = best is not None and p2["period"] is not None and res[(i, best)]["period"] <= p2["period"] * (1 - MIN_IMPROVEMENT)
            rec["fifo_candidates"] = {c: res[(i, c)]["period"] for c in cand}
            over = check_caps(d_cur, interface_delta(d_cur, sa, best)) if gate else []
            if gate and over:
                rec["fifo_rejected"] = f"depth {best} would break the cap(s): {over}"
                log(f"    interface depth {best} REJECTED: would break {over}", flush=True)
                gate = False
            if gate:
                depths[i] = best
                rec["action"] = ("folding + " if step else "") + f"interface depth 2 -> {best}"
                log(f"    interface depth 2 -> {best}: pair {res[(i, best)]['period']} (depth 2: {p2['period']})", flush=True)
            else:
                rec["action"] = ("folding, " if step else "") + "interface growth within the limit does not bring the pair within tolerance / is not > 1 % faster: depth 2 kept"
        else:
            rec["action"] = ("folding, " if step else "") + "UNRESOLVED (still beyond tolerance with an unbounded interface)"
            log(f"    UNRESOLVED: unbounded interface still {rec['deep_mismatch_pct'] if step == 0 else round(100 * mis(pd), 2)}% over the reference", flush=True)
        report.append(rec)
    return blocks, d_cur, depths, report, changes
