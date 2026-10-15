"""Simulate a GIVEN folding (a MILP solution) in the analytical block models + simulators, STARTING from the MILP's own FIFO depths, and size the FIFOs to reach the MILP's rate at the least memory.

    python3 MILP/analytical/net_explicit.py <layer_bits_folding_ARM.json> --out-dir <dir> --tag ARM_simfifo [--workers 6] [--max-iterations 3]

A folding json whose per_layer carries the bilinear decoder's frozen depthwise conv (`<up stage>.main_up.1`: written by finn_milp.py since 2026-10-13, or added by add_frozen_dw_fold.py) builds its up blocks WITH that conv.
The MILP prices its FIFOs in closed form and never simulates the design; the analytical flow (net_fold.py) searches its own folding and simulates every block. This script closes the gap: every block model is fed the MILP's
PE / SIMD / threshold PE as explicit folds (bottleneck.explicit_folds) and the per-site bits of the folding (bits_map), and the simulation is seeded with the MILP's FIFO depths (`seeds`, from the json's intra_block_fifos).
BOTH stages use a PACED source at the MILP's target (not a saturated one, which lets the FIFOs run ahead and hides the rate problem), the same --max-iterations, the same 5 % tolerance and a growth limit of 5 doublings (32x the start):
  1. INTRA-block, every block alone (paced at its own MILP cycles, 3 frames, full control over its FIFOs): per iteration grow the FIFOs that hit capacity until the block sustains its MILP cycles within 5 % and nothing
     deadlocks, then shrink every FIFO to its observed occupancy and check again (bottleneck.iterate_sizing); no FIFO is doubled more than 5 times (2**5 = 32x the depth it starts from, SEED_MAX_MULT). Skip / join FIFOs go to the paced
     run's measured need. A block still slower than its target by more than 5 % is WARNED about (`rate.exceeds_tolerance`): that points at the MILP's model (e.g. the window-fill gap of the dilated convs), not at the FIFOs;
  2. INTER-block, the whole net chained (paced at the MILP's frame cycles, 3 frames, intra-block FIFOs and block simulations FROZEN): per iteration (--max-iterations) a deadlock grows the interface FIFOs that are full, a
     chain slower than the target by more than 5 % grows the interface FIFOs at capacity, grown x4 once per iteration (2 -> 8 -> 32 ...), at most 5 steps, then shrunk to the observed occupancy and re-verified; the grown depths are kept only if the net is > 1 % faster than at depth 2. The deadlock check is part of every run. At the FINAL
     stage every interface is shrunk to its observed peak occupancy regardless of the tolerance, the net is re-simulated and the shrunk depths are handed back only if it does not deadlock and is not slower (0.1 %).
     Nothing intra-block is ever resized here: a chain still beyond 5 % of the MILP target is only WARNED about, with the slowest blocks named as the bottleneck candidates.
Output `layer_bits_folding_<tag>.json` = the MILP json with `intra_block_fifos` (every FIFO of the simulation, in the FINN bridge's role vocabulary, with the MILP's own
depth estimate next to it where the MILP listed the FIFO), `inter_block_fifos`, `block_verification`, `fifo_sim` (what was found / grown) and the FIFO totals; the folding is untouched.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "enet", REPO / "MILP", REPO / "MILP" / "utils", REPO / "MILP" / "analytical"):
    sys.path.insert(0, str(p))

import finn_milp  # noqa: E402
import net_fifo  # noqa: E402
import net_fold  # noqa: E402
from bottleneck import MAX_DOUBLINGS, MIN_IMPROVEMENT, fifo_memory, finalize_fifo_costs  # noqa: E402
from node_names import block_output_name, milp_node  # noqa: E402

WARN_TOLERANCE = 0.05               # a rate more than 5 % past its target is printed as a WARNING (never sizes anything, never fails); not a CLI option
FRAMES = 3
IFACE_GROWTH = 4                  # an inter-block FIFO that is full (deadlock) or at capacity (rate miss) grows x4 per whole-net iteration (2 -> 8 -> 32 -> ...) ...
IFACE_MAX_DEPTH = net_fifo.MIN_DEPTH * IFACE_GROWTH ** MAX_DOUBLINGS      # ... at most 5 growth steps (the default --max-iterations): 2 -> 2048 words
IFACE_MIN_IMPROVEMENT = MIN_IMPROVEMENT                    # the grown depths are kept only when the whole net is more than 1 % faster than at the depth-2 baseline; otherwise depth 2 stays


# ---------------------------------------------------------------------------------------------- MILP entry -> explicit folds
LEGACY_FINAL_BIAS_PE = 1   # PE of the final block's bias add (ChannelwiseOp) for folding jsons from BEFORE the MILP had a `final.bias` node (2026-10-11): the real build's value; newer jsons carry the PE


def explicit_for_block(stage: str, kind: str, per_layer: dict, extra: dict, prev_output: str | None) -> tuple[dict, int, list[str]]:
    """({fold key: (pe, simd) or pe}, F = slowest MILP entry of the block in cycles per frame, warnings). Keys are bottleneck.explicit_folds'."""
    warn: list[str] = []

    def lyr(name):
        v = per_layer[name]
        return (v["pe"], v["simd"]), v.get("thr_pe")

    def xpe(name):
        return extra[name]["pe"]

    f: dict = {}
    for key, thr, leaf in {"reg": (("reduce", "Thr_r", "reduce.0"), ("mid", "Thr_m", "conv"), ("expand", "Thr_e", "expand.0")),
                           "dn": (("reduce", "Thr_r", "reduce.0"), ("mid", "Thr_m", "conv.0"), ("expand", "Thr_e", "expand.0")),
                           "up": (("proj", "Thr_p", "main_proj.0"), ("reduce", "Thr_r", "reduce.0"), ("up", "Thr_u", "up.0"), ("expand", "Thr_e", "expand.0")),
                           "init": (("conv", "Thr_c", "conv"),), "final": ()}[kind]:
        mv, tp = lyr(f"{stage}.{leaf}")
        f[key], f[thr] = mv, tp
    if kind == "final":
        f["final"] = lyr("final")[0]
        if "final.bias" in extra:
            f["Bias"] = xpe("final.bias")
        else:                                               # folding json from before the structural bias node: PE 1 (5 cyc/px; 1,310,720 cycles/frame at 512x512 -> fails there by itself)
            f["Bias"] = LEGACY_FINAL_BIAS_PE
            warn.append(f"{stage}: no final.bias node in the folding json, simulated at the legacy PE {LEGACY_FINAL_BIAS_PE}")
        f["LabelSelect"] = xpe("final.argmax")
    elif kind == "init":
        f.update({"Thr_in": xpe("initial.input_quant"), "Dup": xpe("initial.input_quant.dup"), "Thr_m": xpe("initial.pool_quant"), "Thr_act": xpe("initial.act")})
    else:
        f["Dup"] = xpe(milp_node(stage, kind, "Dup", prev_output)[0])
        f.update({"Thr_s": xpe(f"{stage}.skip_quant"), "Add": xpe(f"{stage}.add"), "Thr_out": xpe(f"{stage}.out_act")})
        if extra[f"{stage}.residual_add"]["pe"] != extra[f"{stage}.out_act"]["pe"]:
            warn.append(f"{stage}: residual_add PE {extra[f'{stage}.residual_add']['pe']} != out_act PE {extra[f'{stage}.out_act']['pe']} (the analytical Thr_out merges them; out_act used)")
        if kind == "dn":
            f["skip_mvau"] = (extra[f"{stage}.skip_pad"]["pe"], extra[f"{stage}.skip_pad"]["simd"])
        if kind == "up" and f"{stage}{net_fold.DW_SUFFIX}" in per_layer:       # the frozen depthwise 3x3 conv (add_frozen_dw_fold.py): its fold, and skip_quant is the depthwise path's threshold Thr_k
            f[f"{stage}{net_fold.DW_SUFFIX}"] = lyr(f"{stage}{net_fold.DW_SUFFIX}")[0]
            f["Thr_k"] = xpe(f"{stage}.skip_quant")
    F = max([v["cycles"] for v in per_layer.values() if v["stage"] == stage] + [v["cycles"] for v in extra.values() if v["stage"] == stage])
    return f, int(F), warn


def seeds_for_block(d: dict, stage: str) -> dict:
    """The MILP's own FIFO depths of one block (its intra_block_fifos): {'skip': ..., 'main': ..., 'prefetch': ...}, each only when the MILP lists it. The simulation STARTS from these
    (verify_with_sim(seeds=...)); a FIFO the MILP does not list starts from the analytic block model's estimate."""
    out = {}
    for f in d.get("intra_block_fifos", []):
        if f.get("stage") != stage:
            continue
        if f.get("is_skip"):
            out["skip"] = int(f["depth"])
        elif f["name"].endswith("FIFO_main"):
            out["main"] = int(f["depth"])
        elif f["consumer"].endswith(".fmpad") or f["name"].endswith(".prefetch"):
            out.setdefault("prefetch", int(f["depth"]))
    dw = d.get("per_layer", {}).get(f"{stage}{net_fold.DW_SUFFIX}")
    if dw and "prefetch_words" in dw:
        out.setdefault("prefetch", int(dw["prefetch_words"]))              # the frozen depthwise conv's prefetch FIFO: not a MILP layer, add_frozen_dw_fold.py's estimate
    return out


def bits_map_for_block(stage: str, kind: str, sites: dict, prev_out: str | None) -> dict:
    """The per-role precision of a block (bottleneck.RoleBits roles) from the PER-SITE bits of the network (layer_bits_SITES: what the network is built and QAT-trained with, so what FINN gets):
    activation roles are the quantizer sites on the block's wires (reduce.2 -> Thr_r, conv_bn_act.2 / conv.2 / up.2 -> Thr_m / Thr_u, residual_add = the add's shared quantizer: both operands and the Add
    input, out_act -> Thr_out), `in` is the previous block's output stream (prev_out: its out_act / initial.act site), weights are the layers' weight sites."""
    wb, ab = sites["layer_weight_bits"], sites["layer_act_bits"]
    S = stage
    if kind == "reg":
        return {"in": ab[prev_out], "r": ab[f"{S}.reduce.2"], "m": ab[f"{S}.conv_bn_act.2"], "e": ab[f"{S}.residual_add"], "s": ab[f"{S}.residual_add"], "add": ab[f"{S}.residual_add"],
                "out": ab[f"{S}.out_act"], "w_r": wb[f"{S}.reduce.0"], "w_m": wb[f"{S}.conv"], "w_e": wb[f"{S}.expand.0"]}
    if kind == "dn":
        return {"in": ab[prev_out], "r": ab[f"{S}.reduce.2"], "m": ab[f"{S}.conv.2"], "e": ab[f"{S}.residual_add"], "s": ab[f"{S}.residual_add"], "add": ab[f"{S}.residual_add"],
                "out": ab[f"{S}.out_act"], "w_r": wb[f"{S}.reduce.0"], "w_m": wb[f"{S}.conv.0"], "w_e": wb[f"{S}.expand.0"]}
    if kind == "up":
        ra = ab[f"{S}.residual_add"]
        return {"in": ab[prev_out], "p": ra, "k": ra, "r": ab[f"{S}.reduce.2"], "u": ab[f"{S}.up.2"], "e": ra, "add": ra, "out": ab[f"{S}.out_act"],
                "w_p": wb[f"{S}.main_proj.0"], "w_r": wb[f"{S}.reduce.0"], "w_u": wb[f"{S}.up.0"], "w_e": wb[f"{S}.expand.0"]}
    if kind == "init":
        act = ab["initial.act"]
        return {"in": ab["initial.input_quant"], "c": act, "m": act, "cat": act, "act": act, "w_c": wb["initial.conv"]}
    return {"in": ab[prev_out], "w_f": wb["final"]}                      # final: the layer after the last out_act


def build_blocks(d: dict, a, ctx) -> tuple[list, list[str]]:
    """[(stage, kind, BottleneckResult built from the MILP folds)], warnings. Asserts that the model landed on exactly the MILP's pe / simd / thr_pe."""
    geoms, extras, geom, xnode, dmap, kinds = ctx
    per_layer, extra = d["per_layer"], d["extra_nodes"]
    blocks, warns, prev = [], [], None
    for stage in net_fold.block_order(geoms):
        kind = net_fold.block_kind(stage)
        folds, F, w = explicit_for_block(stage, kind, per_layer, extra, prev)
        warns += w
        F_milp = F                                           # the MILP's own frame cycles of the block's slowest node: the rate target (the integer T below is only the model's input)
        if kind == "reg":                                   # model_bottleneck takes an integer cycles-per-pixel target
            px = geom[f"{stage}.reduce.0"].hin * geom[f"{stage}.reduce.0"].win
            F = -(-F // px) * px
        bm = bits_map_for_block(stage, kind, a.sites, prev) if getattr(a, "sites", None) else None
        r, lf, xf, _slowest, _fifos, _fw = net_fold.run_block(stage, geom, a.fallback_bits, F, prev_output=prev, explicit=folds, bits_map=bm)
        for name, (pe, simd, thr_pe, _cyc) in lf.items():
            m = per_layer[name]
            assert (pe, simd) == (m["pe"], m["simd"]) and (thr_pe is None or thr_pe == m.get("thr_pe")), (name, (pe, simd, thr_pe), (m["pe"], m["simd"], m.get("thr_pe")))
        for name, (pe, simd, _rs) in xf.items():
            if name in extra and name != f"{stage}.residual_add":
                assert pe == extra[name]["pe"], (name, pe, extra[name]["pe"])
        r.params["F_milp"] = F_milp
        fb = r.params.get("bits_fallback") or []
        if fb:
            w = f"{stage}: roles {fb} are not covered by the folding's per-site bits and use --fallback-bits {a.fallback_bits}"
            print(f"  WARNING (bits fallback): {w}", flush=True)
            warns.append(f"bits: {w}")
        r.params["seeds"] = seeds_for_block(d, stage)
        blocks.append((stage, kind, r))
        prev = block_output_name(stage, kind)
    return blocks, warns


# ---------------------------------------------------------------------------------------------- per-block simulation (parallel)
def _verify_keep(args):
    """verify_with_sim(seeds=...): the sizing STARTS from the MILP's FIFO depths (the analytic estimate where the MILP lists none) and may raise or lower any FIFO to reach the block's MILP rate: skip / join
    FIFOs go to the paced run's measured need; every other FIFO is sized iteratively (bottleneck.iterate_sizing, max_it rounds of grow-the-FIFOs-at-capacity -> check cycles + deadlock -> shrink to the observed
    occupancy -> check again), none past SEED_MAX_MULT (6, +500 %) times its start. Fails only on a DEADLOCK; a rate beyond the tolerance is recorded (annotate_rate) and warned about."""
    kind, r, stage, tol, max_it = args
    import bottleneck, dn_bottleneck, fnl_block, int_bottleneck, up_bottleneck
    ver = {"reg": bottleneck, "dn": dn_bottleneck, "up": up_bottleneck, "init": int_bottleneck, "final": fnl_block}[kind].verify_with_sim
    try:
        ver(r, tol=tol, tol_soft=tol, deadlock_only=True, seeds=r.params.get("seeds", {}), reach_tol=tol, max_iterations=max_it)
    except RuntimeError as e:
        return r, str(e)
    return r, None


def annotate_rate(blocks: list, tol: float, f_milp: dict | None = None) -> None:
    """r.verification['rate'] for EVERY block (also cached ones): the target (the MILP's own cycles per frame of the block's slowest node, f_milp[stage] or r.params['F_milp'], as cycles per output
    pixel; r.params['T'] -- the integer the regular model is built with, rounded UP -- only when neither is known), the achieved steady rate, the block's frame period and whether it is beyond the
    tolerance. `rate_miss` (the pre-2026-10-12 key, read by summarize_simfifo.py) is kept for the blocks beyond it. Idempotent."""
    for stage, _, r in blocks:
        v = r.verification
        F = (f_milp or {}).get(stage) or r.params.get("F_milp")
        T, ach = (F / (r.params["hout"] * r.params["wout"]) if F else r.params["T"]), v["steady_cyc_px"]
        over = ach / T - 1
        v["rate"] = dict(target_cyc_px=T, achieved_cyc_px=ach, over_pct=round(100 * over, 2), period_cycles=int(ach * r.params["hout"] * r.params["wout"]), tolerance_pct=round(100 * tol, 2),
                         exceeds_tolerance=bool(over > tol + 1e-9))
        v.pop("rate_miss", None)
        if v["rate"]["exceeds_tolerance"]:
            v["rate_miss"] = dict(v["rate"])


def deadlock_errors(blocks: list, errors: dict) -> dict:
    """Keep only the failures that are deadlocks (cached block results from before the single-tolerance rule also carry rate-only failures)."""
    by_stage = {stage: r for stage, _, r in blocks}
    return {stage: err for stage, err in errors.items() if by_stage[stage].verification.get("deadlock")}


def verify_blocks_keep(blocks: list, workers: int, tol: float, max_iterations: int = 3) -> tuple[list, dict]:
    key = lambda kind, r: repr((kind, sorted(r.params.items()), [(n.name, n.pe, n.simd) for n in r.nodes]))
    todo: dict = {}
    for stage, kind, r in blocks:
        todo.setdefault(key(kind, r), (kind, r, stage, tol, max_iterations))
    print(f"simulating {len(blocks)} blocks ({len(todo)} distinct shapes), {workers} workers", flush=True)
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            done = pool.map(_verify_keep, list(todo.values()))
    else:
        done = [_verify_keep(t) for t in todo.values()]
    ver = dict(zip(todo, done))
    out, errors = [], {}
    for stage, kind, r in blocks:
        r2, err = ver[key(kind, r)]
        out.append((stage, kind, copy.deepcopy(r2)))            # blocks of one shape share the result: each gets its own copy (the chain grows them individually)
        if err:
            errors[stage] = err
    return out, errors


# ---------------------------------------------------------------------------------------------- growing FIFOs
def grow_intra(r, name: str, new_depth: int) -> int:
    """Set the depth of intra-block FIFO `name` in the verified graph (and the skip FIFO record); returns the old depth."""
    f = r.fifo_graph["fifos"][name]
    old = int(f["depth"])
    f["depth"] = int(new_depth)
    if name == "skip FIFO" and r.skip_fifo is not None:
        sf = r.skip_fifo
        words_px = r.params["cout"] // sf.pe
        sf.depth_words, sf.bits, sf.pixels_buffered = int(new_depth), sf.width_bits * int(new_depth), math.ceil(new_depth / words_px)
    return old


def _input_pixels(r) -> int:
    return int(r.params["height"] * r.params["width"])


def _chain_once(blocks: list, depths: list[int], F_target: int, tol: float, interval: float, rnd: int) -> tuple[dict, dict]:
    """One whole-net chain run at the given interface depths: the first block's source PACED at `interval` cycles per input pixel (= the MILP's frame cycles / input pixels), the intra-block FIFOs and the
    block simulations as they are (frozen). Returns (raw result, info): deadlock, period, within tolerance of F_target, peak occupancy per interface, the interfaces / intra FIFOs found full."""
    t0 = time.time()
    caps = [net_fifo.capture_block(kind, r, FRAMES, interval if i == 0 else 0) for i, (_, kind, r) in enumerate(blocks)]
    owner = {}
    for i, b in enumerate(caps):
        for name, f in b["fifos"].items():
            owner[id(f)] = (i, name)
    net = net_fifo.compose(caps, depths)
    res = net_fifo.run_chain(net, max_cycles=int(FRAMES * F_target * 3) + 400000)
    iface = {id(f): j for j, f in enumerate(net["iface"])}
    period = None if res["period"] == float("inf") else int(res["period"])
    limit = F_target * (1 + tol)
    ok = not res["deadlock"]

    def label(f):
        if id(f) in iface:
            return f"iface{iface[id(f)]} ({blocks[iface[id(f)]][0]} -> {blocks[iface[id(f)] + 1][0]})"
        i, name = owner.get(id(f), (None, f.name))
        return f"{blocks[i][0]}:{name}" if i is not None else f.name

    info = dict(round=rnd, deadlock=res["deadlock"], period=period, within_tolerance=bool(ok and res["period"] <= limit), vs_target_pct=(None if period is None else round(100 * (period / F_target - 1), 2)),
                first_out=res["first_out"], interface_depths=list(depths), iface_max=res["iface_max"], full=[label(f) for f in res["full"]],
                full_interfaces=sorted(iface[id(f)] for f in res["full"] if id(f) in iface), seconds=round(time.time() - t0))
    return res, info


def choose_interface_state(baseline: dict, states: list[dict], min_improvement: float) -> dict:
    """The depths to hand back: the best (lowest period, then fewest words) deadlock-free state that went through a growth AND is more than `min_improvement` faster than the baseline; the baseline
    (depth 2) when there is none. A deadlocked baseline has nothing to beat: every deadlock-free grown state qualifies."""
    ok = [i for i in states if i.get("grown") and not i["deadlock"] and i["period"] is not None
          and (baseline["deadlock"] or baseline["period"] is None or i["period"] <= baseline["period"] * (1 - min_improvement))]
    if not ok:
        return baseline
    return min(ok, key=lambda i: (i["period"], sum(i["interface_depths"])))


def size_interfaces(blocks: list, F_target: int, tol: float, max_iterations: int, log=print) -> dict:
    """INTER-block FIFO sizing on the whole net, the same loop as the intra-block sizing (bottleneck.iterate_sizing) on the interface FIFOs. The source is paced at F_target frame cycles (the MILP's own
    target); nothing intra-block is resized. Start: every interface at net_fifo.MIN_DEPTH. A baseline run first, then per iteration (at most max_iterations, default 5):
      1. ONE GROWTH STEP when the chain deadlocks (the full interfaces) or is slower than F_target * (1 + tol) (the interfaces whose peak occupancy reached their depth): x IFACE_GROWTH (4: 2 -> 8 -> 32 ...),
         at most MAX_DOUBLINGS (5) steps; run again;
      2. SHRINK every interface to its observed peak occupancy (>= net_fifo.MIN_DEPTH) and run again: accepted when it does not deadlock and is no slower than the run it shrinks (0.1 %) or is within the
         tolerance -- so what is handed back has always been deadlock-checked, tolerance met or not.
    Ends early when an iteration changes nothing. The depths handed back (choose_interface_state): the best grown state if it is more than 1 % faster than the depth-2 baseline, else the BASELINE (growth that does
    not pay is discarded). `ok` = they are deadlock-free. `within_tolerance` = their period is within F_target * (1 + tol); if not, the caller warns (the intra-block FIFOs are not touched here) and names the slowest blocks."""
    n = len(blocks)
    interval = F_target / _input_pixels(blocks[0][2])
    depths = [net_fifo.MIN_DEPTH] * (n - 1)
    limit = F_target * (1 + tol)
    rounds, grown, shrinks = [], {}, []

    def run(depths_now, phase, is_grown):
        _, info = _chain_once(blocks, depths_now, F_target, tol, interval, len(rounds) + 1)
        info["phase"], info["grown"] = phase, is_grown
        rounds.append(info)
        log(f"  inter-block {phase}: deadlock={info['deadlock']} period={info['period']} ({info['vs_target_pct']}% vs the MILP target {F_target}, limit {limit:.0f}, within={info['within_tolerance']}) "
            f"max depth {max(depths_now)}, sum {sum(depths_now)}, {len(info['full'])} full FIFO(s) [{info['seconds']} s]", flush=True)
        return info

    baseline = cur = run(depths, "baseline (depth 2)", False)
    went_up = False
    for it in range(1, max_iterations + 1):
        changed = False
        if cur["deadlock"] or not cur["within_tolerance"]:
            if cur["deadlock"]:
                targets = [j for j in cur["full_interfaces"] if depths[j] < IFACE_MAX_DEPTH]
                for name in cur["full"]:
                    if not name.startswith("iface"):
                        log(f"    full intra-block FIFO {name} (not resized here)", flush=True)
            else:
                targets = [j for j, m in enumerate(cur["iface_max"]) if m >= depths[j] and depths[j] < IFACE_MAX_DEPTH]
            if targets:
                for j in targets:
                    old = depths[j]
                    depths[j] = min(old * IFACE_GROWTH, IFACE_MAX_DEPTH)
                    grown.setdefault(j, [old, old])[1] = depths[j]
                went_up = True
                cur = run(depths, f"iteration {it} grow {len(targets)} interface(s) x{IFACE_GROWTH}", True)
                cur["grown_interfaces"] = [f"iface{j} ({blocks[j][0]} -> {blocks[j + 1][0]})" for j in targets]
                changed = True
            else:
                log("    no full / at-capacity interface below its growth limit -- nothing to grow", flush=True)
        if not cur["deadlock"]:
            small = [max(net_fifo.MIN_DEPTH, min(d, m)) for d, m in zip(depths, cur["iface_max"])]
            if small != depths:
                info = run(small, f"iteration {it} shrink to the observed occupancy", went_up)
                accepted = (not info["deadlock"]) and info["period"] is not None and (info["period"] <= cur["period"] * 1.001 or info["within_tolerance"])
                shrinks.append(dict(iteration=it, accepted=bool(accepted), period=info["period"], deadlock=info["deadlock"], sum_before=sum(depths), sum_after=sum(small)))
                log(f"    shrink {'KEPT' if accepted else 'rejected, previous depths kept'} (sum {sum(depths)} -> {sum(small) if accepted else sum(depths)})", flush=True)
                if accepted:
                    depths, cur = small, info
                    changed = True
        if not changed:
            break
    chosen = choose_interface_state(baseline, rounds, IFACE_MIN_IMPROVEMENT)
    discarded = chosen is baseline and went_up
    if discarded:
        best_p = min((i["period"] for i in rounds if i.get("grown") and not i["deadlock"] and i["period"] is not None), default=None)
        log(f"  growth DISCARDED: the best grown state ({best_p}) is not more than {100 * IFACE_MIN_IMPROVEMENT:g}% faster than the depth-2 baseline ({baseline['period']}) -> depth 2 kept", flush=True)
    ok = not chosen["deadlock"]
    slowest = sorted(((own_period(r), stage) for stage, _, r in blocks), reverse=True)[:3]
    return dict(ok=ok, within_tolerance=bool(ok and chosen["within_tolerance"]), rounds=rounds, interface_depths=list(chosen["interface_depths"]), growth_discarded=bool(discarded),
                min_improvement_pct=round(100 * IFACE_MIN_IMPROVEMENT, 2), baseline_period=baseline["period"],
                grown=[dict(where="iface", fifo=f"iface{j}", depth_before=v[0], depth_after=v[1]) for j, v in grown.items()], shrink=shrinks, last=chosen, target=F_target, limit=limit, milp_cap=F_target,
                tolerance_pct=round(100 * tol, 2), pace_cycles_per_input_px=round(interval, 4), growth=IFACE_GROWTH, max_depth=IFACE_MAX_DEPTH, max_iterations=max_iterations,
                bottleneck_candidates=[dict(block=stage, period_cycles=p) for p, stage in slowest])


def own_period(r) -> int:
    """The block's own simulated frame period (annotate_rate's value when it ran, else steady cyc/px x output pixels)."""
    v = r.verification
    return int(v["rate"]["period_cycles"]) if "rate" in v else int(v["steady_cyc_px"] * r.params["hout"] * r.params["wout"])


# ---------------------------------------------------------------------------------------------- main / output
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folding_json", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="output file tag: layer_bits_folding_<tag>.json")
    ap.add_argument("--config", default="config_12_dense_relu_nearest_upsample_256")
    ap.add_argument("--fallback-bits", type=int, default=8, help="precision (weights and activations) of a role the folding's per-site bits do not cover (default 8); every block that has to use it is WARNED about. With --uniform-bits it is the precision of every role")
    ap.add_argument("--uniform-bits", action="store_true", help="ignore the folding's per-site bits and model every stream and weight at --fallback-bits (the pre-2026-10-11 behaviour)")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--max-iterations", type=int, default=5, help="sizing iterations of BOTH stages (default 5). Intra-block (per block, paced source): grow the FIFOs at capacity until the block's MILP cycles are reached within 5 %% and nothing deadlocks, then shrink to the observed occupancy and check again (bottleneck.iterate_sizing, at most 5 doublings = 32x the starting depth). Inter-block (whole net, paced at the MILP target): grow the full / at-capacity interface FIFOs x4 (2 -> 8 -> 32 ..., at most 5 steps), shrink + verify every iteration; growth that does not make the net > 1 %% faster is discarded (depth 2 stays) until the chain is deadlock-free and within 5 %%; a final shrink + deadlock check always follows")
    ap.add_argument("--reuse-blocks", action="store_true", help="reuse <out-dir>/blocks.pkl (the simulated blocks of an earlier run) instead of re-simulating every block")
    ap.add_argument("--only", default="", help="with --reuse-blocks: comma-separated stages to re-simulate (e.g. up4,up5); every other block is taken from the cache")
    ap.add_argument("--no-chain", action="store_true", help="per-block simulation only (skip the whole-net runs)")
    a = ap.parse_args()
    raise SystemExit(run(a))


def run(a) -> int:
    from bottleneck import WWIDTH_MAX  # noqa: F401  (explicit folds skip the width cap)
    d = json.loads(a.folding_json.read_text())
    assert d["status"] == "Optimal", d["status"]
    finn_milp.load_config(a.config)
    a.sites = None if a.uniform_bits else net_fold.sites_for(a.config, a.folding_json)           # per-site bits of the network -> per-block bits_map (bottleneck.RoleBits)
    finn_milp.CANDIDATE_BITS = tuple(sorted(set(finn_milp.CANDIDATE_BITS) | {a.fallback_bits}))
    model, geoms, extras, _pred, dmap, kinds = finn_milp.build_model_and_graph()
    if any(n.endswith(net_fold.DW_SUFFIX) for n in d["per_layer"]):          # a folding json from add_frozen_dw_fold.py: the up blocks carry the bilinear decoder's frozen depthwise 3x3 conv
        net_fold.BILINEAR_UP = True
        added = net_fold.add_bilinear_dw(geoms, dmap)
        print(f"{a.tag}: up blocks built WITH the frozen depthwise conv ({', '.join(added)})", flush=True)
    ctx = (geoms, extras, {g.name: g for g in geoms}, {n.geom.name: n for n in extras}, dmap, kinds)
    t0 = time.time()
    blocks, warns = build_blocks(d, a, ctx)
    f_milp = {stage: r.params["F_milp"] for stage, _, r in blocks}
    print(f"{a.tag}: {len(blocks)} blocks built from the MILP folding ({len(warns)} warnings)", flush=True)
    tol = WARN_TOLERANCE
    milp_cap = int(d["_diagnostics"]["max_node_cycles"])
    cache = a.out_dir / "blocks.pkl"
    if a.reuse_blocks and cache.exists():
        cached, cerrors = pickle.loads(cache.read_bytes())
        if a.only:                                       # re-simulate just these blocks, take the others from the cache (their folding is unchanged)
            only = set(a.only.split(","))
            fresh, ferrors = verify_blocks_keep([b for b in blocks if b[0] in only], a.workers, tol, a.max_iterations)
            fm = {b[0]: b for b in fresh}
            blocks = [fm.get(b[0], b) for b in cached]
            errors = {**{k: v for k, v in cerrors.items() if k not in only}, **ferrors}
            cache.write_bytes(pickle.dumps((blocks, errors)))
            print(f"re-simulated {sorted(only)}, reusing the other {len(blocks) - len(fm)} blocks of {cache}", flush=True)
        else:
            blocks, errors = cached, cerrors
            print(f"reusing the simulated blocks of {cache}", flush=True)
    else:
        blocks, errors = verify_blocks_keep(blocks, a.workers, tol, a.max_iterations)
        a.out_dir.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps((blocks, errors)))
    errors = deadlock_errors(blocks, errors)
    annotate_rate(blocks, tol, f_milp)
    for stage, err in errors.items():
        print(f"  BLOCK DEADLOCKS ALONE: {stage}: {err}", flush=True)
    for stage, _, r in blocks:
        if r.verification["rate"]["exceeds_tolerance"]:
            rt = r.verification["rate"]
            msg = f"{stage} reaches {rt['achieved_cyc_px']:.2f} vs its target {rt['target_cyc_px']:.2f} cyc/px ({rt['over_pct']:+.2f}% > the {rt['tolerance_pct']:g}% tolerance), frame period {rt['period_cycles']}; FIFOs sized deadlock-free only"
            print(f"  WARNING (rate, not a failure): {msg}", flush=True)
            warns.append(f"rate: {msg}")
    nodes = {**d["per_layer"], **d["extra_nodes"]}
    F_net = int(max([v["cycles"] for v in nodes.values()] + [r.verification["rate"]["period_cycles"] for _, _, r in blocks]))
    chain = None
    if not a.no_chain:
        print(f"whole-net chain: source PACED at the MILP target {milp_cap} frame cycles, up to {a.max_iterations} iteration(s), interfaces x{IFACE_GROWTH} per iteration (<= {MAX_DOUBLINGS} steps, depth <= {IFACE_MAX_DEPTH}) with a shrink + deadlock check each iteration, growth kept only if > {100 * IFACE_MIN_IMPROVEMENT:g}% faster than depth 2; "
              f"tolerance {100 * tol:g}% (intra-block FIFOs frozen)", flush=True)
        chain = size_interfaces(blocks, milp_cap, tol, a.max_iterations)
        if not chain["ok"]:
            msg = "whole-net chain DEADLOCKS at every iteration (only the inter-block FIFOs were resized): " + "; ".join(chain["last"]["full"][:6])
            print(f"  FAILED: {msg}", flush=True)
            warns.append(f"chain: {msg}")
        elif not chain["within_tolerance"]:
            lr = chain["last"]
            names = ", ".join(f"{b['block']} ({b['period_cycles']})" for b in chain["bottleneck_candidates"])
            msg = (f"whole-net period {lr['period']} is {lr['vs_target_pct']:+.2f}% over the MILP target {milp_cap} (tolerance {100 * tol:g}%); deadlock-free. The inter-block FIFOs are at their limit; "
                   f"slowest blocks (bottleneck candidates, own period in cycles): {names}")
            print(f"  WARNING (rate, not a failure): {msg}", flush=True)
            warns.append(f"rate: {msg}")
    out = assemble_output(d, blocks, chain, errors, warns, a, F_net, time.time() - t0)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    path = a.out_dir / f"layer_bits_folding_{a.tag}.json"
    path.write_text(json.dumps(out, indent=1))
    rep = out["fifo_sim"]
    print(f"{a.tag}: wrote {path}\n  blocks deadlocking alone: {len(errors)}, chain deadlock-free: {rep['chain_ok']}, within tolerance: {rep['chain_within_tolerance']}, intra FIFO BRAM18 {rep['bram18_milp_estimate']} (MILP) -> {rep['bram18_sim']} (sim)", flush=True)
    return 0 if (not errors and (chain is None or chain["ok"])) else 2


def assemble_output(d: dict, blocks: list, chain: dict | None, errors: dict, warns: list[str], a, F_net: int, seconds: float) -> dict:
    out = copy.deepcopy(d)
    g = out["_diagnostics"]
    milp_depth = {(f["stage"], f["producer"], f["consumer"]): f["depth"] for f in d.get("intra_block_fifos", [])}
    fm = g.pop("fifo_model", None)
    fm_tot = (fm or {}).get("totals", {})
    inter_old = d.get("inter_block_fifos", [])
    # edge_fifo_lut = every depth-2 edge FIFO of the MILP, inter-block ones included (finn_milp adds it to total_lut_calibrated); older jsons have no such key and list the inter-block LUT separately
    est_lut = fm_tot.get("lut", 0) + fm_tot.get("dwc_lut", 0) + (fm_tot["edge_fifo_lut"] if "edge_fifo_lut" in fm_tot else sum(f.get("lut", 0) for f in inter_old))
    est_bram = fm_tot.get("bram18", 0) + sum(f.get("bram18", 0) for f in inter_old)
    # node-only totals (analytical convention: FIFOs and DWCs on top)
    g["total_lut_calibrated"] -= est_lut
    g["total_bram18k_calibrated"] -= est_bram
    g["fifo_model_estimate"] = dict(fifo_model=fm, lut=est_lut, bram18=est_bram, note="the MILP's closed-form FIFO / DWC estimate that was removed from total_lut_calibrated / total_bram18k_calibrated (now nodes only)")
    # inter-block FIFOs: grown depths from the chain
    depths = chain["interface_depths"] if chain else [net_fold_min() for _ in range(len(blocks) - 1)]
    by_src = {f["src_block"]: f for f in out["inter_block_fifos"]}
    for i, dep in enumerate(depths):
        e = by_src.get(blocks[i][0])
        if e is None:
            continue
        m = fifo_memory(e["width_bits"], dep)
        e.pop("aspect", None)
        e.update(depth=dep, mem=m["mem"], depth_alloc=int(m["depth_alloc"]), bram18=int(m["bram18"]), lut=int(m["lut"]), uram18=int(m["uram"]), efficiency=round(m["efficiency"], 3),
                 sizing=("fixed" if dep == net_fold_min() else "sim_grown"))
    ol = out["inter_block_fifos"]
    g["inter_block_fifo_bram18"] = sum(f["bram18"] for f in ol)
    g["inter_block_fifo_lut"] = sum(f.get("lut", 0) for f in ol)
    g["inter_block_fifo_uram18"] = sum(f.get("uram18", 0) for f in ol)
    for _, _, r in blocks:
        finalize_fifo_costs(r)
    holder: dict = {"_diagnostics": g}
    net_fold.intra_block_report(blocks, holder)
    flat = holder["intra_block_fifos"]
    for f in flat:
        f["milp_depth"] = milp_depth.get((f["stage"], f["producer"], f["consumer"]))
    out["intra_block_fifos"], out["block_verification"] = flat, holder["block_verification"]
    g["intra_block_fifo_totals"], g["intra_block_fifo_bram18"] = holder["_diagnostics"]["intra_block_fifo_totals"], holder["_diagnostics"]["intra_block_fifo_bram18"]
    g["total_bram18k_with_all_fifos"] = g["total_bram18k_calibrated"] + g["inter_block_fifo_bram18"] + g["intra_block_fifo_bram18"]
    it = g["intra_block_fifo_totals"]
    g["total_lut_with_all_fifos"] = g["total_lut_calibrated"] + it["fifo_lut"] + it["dwc_lut"] + g["inter_block_fifo_lut"]
    out["fifo_sim"] = dict(
        source=str(a.folding_json.name), tag=a.tag, tolerance_pct=round(100 * WARN_TOLERANCE, 2), fallback_bits=a.fallback_bits, bits_source=("per-site bits of the folding (bits_map per block, bottleneck.RoleBits)" if getattr(a, "sites", None) else "uniform --fallback-bits"), seconds=round(seconds), net_target_cycles=F_net, workers=a.workers,
        blocks={stage: dict(kind=kind, ok=(stage not in errors), error=errors.get(stage), tries=r.verification.get("tries"), deadlock=r.verification.get("deadlock"), T=r.params.get("T"),
                            steady_cyc_px=r.verification.get("steady_cyc_px"), rate=r.verification.get("rate"), rate_miss=r.verification.get("rate_miss"), uniform_depth=r.verification.get("uniform_depth"), elastic_depth=r.verification.get("elastic_depth"), seeds=r.params.get("seeds"),
                            full_fifos=(r.verification.get("full_fifos") or []) if stage in errors else [], warnings=list(r.warnings))
                for stage, kind, r in blocks},
        chain=chain, chain_ok=(None if chain is None else chain["ok"]), chain_within_tolerance=(None if chain is None else chain["within_tolerance"]), build_warnings=warns,
        bram18_milp_estimate=est_bram, bram18_sim=g["intra_block_fifo_bram18"] + g["inter_block_fifo_bram18"],
        note="folding identical to the source MILP json; FIFO depths = the smallest per-block depths that sustain each block's own slowest MILP node in simulation, grown further by the whole-net chain on a deadlock",
    )
    return out


def net_fold_min() -> int:
    return net_fifo.MIN_DEPTH


if __name__ == "__main__":
    main()
