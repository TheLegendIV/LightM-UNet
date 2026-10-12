"""Simulate a GIVEN folding (a MILP solution) in the analytical block models + simulators, STARTING from the MILP's own FIFO depths, and size the FIFOs to reach the MILP's rate at the least memory.

    python3 MILP/analytical/net_explicit.py <layer_bits_folding_ARM.json> --out-dir <dir> --tag ARM_simfifo [--workers 6] [--max-iterations 3] [--max-rounds 1]

The MILP prices its FIFOs in closed form (skip, prefetch, up-block join, the 32-word down / up block FIFOs, depth-2 edges) and never simulates the design; the analytical flow (net_fold.py) searches its own
folding and simulates every block. This script closes the gap: every block model is fed the MILP's PE / SIMD / threshold PE as explicit folds (bottleneck.explicit_folds) and the per-site bits of the folding
(bits_map), and the simulation is seeded with the MILP's FIFO depths (`seeds`, from the json's intra_block_fifos; the analytic estimate where the MILP lists none). Then
  1. each block is simulated alone (saturated input, 3 frames) with FULL CONTROL over its FIFOs, each limited to +500 % (6x) of its starting depth. Skip / join FIFOs go to the paced run's measured need.
     Every other FIFO is sized ITERATIVELY (bottleneck.iterate_sizing, --max-iterations rounds, default 3): grow the FIFOs that hit their capacity until the block reaches its MILP cycles within
     WARN_TOLERANCE (5 %) and nothing deadlocks -> shrink everything to its observed occupancy and check cycles, deadlock and the paced run again -> repeat; the cheapest state that reaches the target
     is kept, else the closest. A block still slower than its target by more than the tolerance is WARNED about (`rate.exceeds_tolerance`): that points at a shortfall in the MILP's model (e.g. the
     window-fill gap of the dilated convs), not at the FIFOs;
  2. the whole network is chained (inter-block FIFOs at depth 2, saturated input, 3 frames). Pass = no deadlock. The last-frame period is compared with the slowest block's own
     achieved period within the same tolerance and reported (`within_tolerance`). Only a DEADLOCK makes the full FIFOs grow and the chain re-run, up to --max-rounds (default 1: a pure check,
     nothing grown unverified).
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
from bottleneck import fifo_memory, finalize_fifo_costs  # noqa: E402
from node_names import block_output_name, milp_node  # noqa: E402

WARN_TOLERANCE = 0.05               # a rate more than 5 % past its target is printed as a WARNING (never sizes anything, never fails); not a CLI option
MAX_TRIES = 10                     # the extended _SIZING_SCHEDULE has 10 entries (up/int/fnl default to 6)
FRAMES = 3
INTRA_GROWTH = 4                  # a full intra-block FIFO of a deadlocked chain is grown x4 per round (every round of the real nets costs minutes); interfaces x2


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
        r, lf, xf, _slowest, _fifos, _fw = net_fold.run_block(stage, geom, a.bits, F, prev_output=prev, explicit=folds, bits_map=bm)
        for name, (pe, simd, thr_pe, _cyc) in lf.items():
            m = per_layer[name]
            assert (pe, simd) == (m["pe"], m["simd"]) and (thr_pe is None or thr_pe == m.get("thr_pe")), (name, (pe, simd, thr_pe), (m["pe"], m["simd"], m.get("thr_pe")))
        for name, (pe, simd, _rs) in xf.items():
            if name in extra and name != f"{stage}.residual_add":
                assert pe == extra[name]["pe"], (name, pe, extra[name]["pe"])
        r.params["F_milp"] = F_milp
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
        ver(r, max_tries=MAX_TRIES, tol=tol, tol_soft=tol, deadlock_only=True, seeds=r.params.get("seeds", {}), reach_tol=tol, max_iterations=max_it)
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


def run_chain_rounds(blocks: list, F_net: int, tol: float, max_rounds: int, log=print, milp_cap: int | None = None) -> dict:
    """Chain the blocks (inter-block depth 2) and, on a deadlock, grow the full FIFOs and re-run, up to max_rounds. `ok` = no deadlock (the pass criterion). `within_tolerance` = the last-frame
    period is within F_net * (1 + tol), F_net = the slowest block's own achieved period (or the MILP node): a rate miss beyond it is only reported (FIFOs do not change the steady rate).
    Nothing is grown after the LAST round (it would not be verified). Mutates the blocks' fifo_graph."""
    n = len(blocks)
    depths = [net_fifo.MIN_DEPTH] * (n - 1)
    rounds, grown = [], {}                          # grown: (stage, fifo name) | ("iface", i) -> [first depth, last depth]
    limit = F_net * (1 + tol)
    ok = within = False
    for rnd in range(1, max_rounds + 1):
        t0 = time.time()
        caps = [net_fifo.capture_block(kind, r, FRAMES) for _, kind, r in blocks]
        owner = {}
        for i, b in enumerate(caps):
            for name, f in b["fifos"].items():
                owner[id(f)] = (i, name)
        net = net_fifo.compose(caps, depths)
        res = net_fifo.run_chain(net, max_cycles=int(FRAMES * F_net * 3) + 400000)
        iface = {id(f): j for j, f in enumerate(net["iface"])}
        ok = not res["deadlock"]
        within = ok and res["period"] <= limit

        def label(f):
            if id(f) in iface:
                return f"iface{iface[id(f)]} ({blocks[iface[id(f)]][0]} -> {blocks[iface[id(f)] + 1][0]})"
            i, name = owner.get(id(f), (None, f.name))
            return f"{blocks[i][0]}:{name}" if i is not None else f.name
        period = None if res["period"] == float("inf") else int(res["period"])
        info = dict(round=rnd, deadlock=res["deadlock"], period=period, within_tolerance=bool(within),
                    vs_slowest_block_pct=(None if period is None else round(100 * (period / F_net - 1), 2)), vs_milp_cap_pct=(None if period is None or not milp_cap else round(100 * (period / milp_cap - 1), 2)),
                    first_out=res["first_out"], iface_max=res["iface_max"], full=[label(f) for f in res["full"]], seconds=round(time.time() - t0))
        rounds.append(info)
        log(f"  chain round {rnd}: deadlock={res['deadlock']} period={period} (slowest block {F_net}, tolerance limit {limit:.0f}, within={within}) first_out={info['first_out']} full={len(info['full'])} [{info['seconds']} s]", flush=True)
        if ok or rnd == max_rounds:                  # no deadlock = done: a rate miss is reported, FIFOs do not fix it
            break
        targets = res["full"]
        acted = []
        for f in targets:
            if id(f) in iface:
                j = iface[id(f)]
                old = depths[j]
                depths[j] = old * 2
                grown.setdefault(("iface", j), [old, old])[1] = depths[j]
                acted.append(label(f))
            elif id(f) in owner:
                i, name = owner[id(f)]
                g = blocks[i][2].fifo_graph["fifos"][name]
                if g["producer"] == "Source" or g["consumer"] == "Sink":
                    continue
                old = grow_intra(blocks[i][2], name, INTRA_GROWTH * int(g["depth"]))
                grown.setdefault((blocks[i][0], name), [old, old])[1] = INTRA_GROWTH * old
                acted.append(label(f))
        info["grown"] = acted
        if not acted:
            log("  nothing to grow: stuck", flush=True)
            break
        for _, _, r in blocks:
            finalize_fifo_costs(r)
    return dict(ok=ok, within_tolerance=bool(within), rounds=rounds, interface_depths=depths,
                grown=[dict(where=k[0], fifo=(k[1] if k[0] != "iface" else f"iface{k[1]}"), depth_before=v[0], depth_after=v[1]) for k, v in grown.items()],
                last=rounds[-1] if rounds else None, target=F_net, limit=limit, milp_cap=milp_cap, tolerance_pct=round(100 * tol, 2))


# ---------------------------------------------------------------------------------------------- main / output
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folding_json", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="output file tag: layer_bits_folding_<tag>.json")
    ap.add_argument("--config", default="config_12_dense_relu_nearest_upsample_256")
    ap.add_argument("--bits", type=int, default=6, help="uniform fallback precision; with the default (per-site bits) it only fills the roles a block has no site for")
    ap.add_argument("--uniform-bits", action="store_true", help="ignore the folding's per-site bits and model every stream and weight at --bits (the pre-2026-10-11 behaviour)")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--max-iterations", type=int, default=3, help="per-block sizing iterations (default 3): each = grow the FIFOs at capacity until the block's MILP cycles are reached within 5 %% and nothing deadlocks, then shrink every FIFO to its observed occupancy and check cycles + deadlock again (bottleneck.iterate_sizing)")
    ap.add_argument("--max-rounds", type=int, default=1, help="whole-net chain rounds (default 1: one pure check). More rounds grow the full FIFOs of a deadlocked chain (x4 intra, x2 interfaces) and re-run")
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
    finn_milp.CANDIDATE_BITS = tuple(sorted(set(finn_milp.CANDIDATE_BITS) | {a.bits}))
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
        print(f"whole-net chain: pass = no deadlock; period reported against the slowest block {F_net} (+{100 * tol:g}%) and the MILP cap {milp_cap}; inter-block depth 2, up to {a.max_rounds} round(s)", flush=True)
        chain = run_chain_rounds(blocks, F_net, tol, a.max_rounds, milp_cap=milp_cap)
        if chain["ok"] and not chain["within_tolerance"]:
            lr = chain["last"]
            msg = f"whole-net period {lr['period']} is {lr['vs_slowest_block_pct']:+.2f}% over the slowest block's own {F_net} (tolerance {100 * tol:g}%), {lr['vs_milp_cap_pct']:+.2f}% over the MILP cap {milp_cap}; deadlock-free"
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
        source=str(a.folding_json.name), tag=a.tag, tolerance_pct=round(100 * WARN_TOLERANCE, 2), bits=a.bits, bits_source=("per-site bits of the folding (bits_map per block, bottleneck.RoleBits)" if getattr(a, "sites", None) else "uniform --bits"), seconds=round(seconds), net_target_cycles=F_net, workers=a.workers,
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
