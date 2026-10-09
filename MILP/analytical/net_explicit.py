"""Simulate a GIVEN folding (a MILP solution) in the analytical block models + simulators and grow its FIFOs until nothing deadlocks.

    python3 MILP/analytical/net_explicit.py <layer_bits_folding_ARM.json> --out-dir <dir> --tag ARM_simfifo [--workers 6]

The MILP prices a few FIFOs in closed form (skip, prefetch, DWC) and never simulates the design; the analytical flow (net_fold.py) searches its own folding
and simulates every block. This script closes the gap: every block model is fed the MILP's PE / SIMD / threshold PE as explicit folds (bottleneck.explicit_folds),
then
  1. each block is simulated alone (saturated input, 3 frames) with its own cycle budget F = the slowest MILP node of the block, and its internal FIFOs are sized
     to the smallest memory that sustains that rate (the usual verify_with_sim schedule, extended to uniform depth x16);
  2. the whole network is chained (inter-block FIFOs at depth 2, saturated input, 3 frames). On a deadlock, the FIFOs that are full are doubled (intra-block ones in
     their block, inter-block ones by index) and the chain is re-run, up to --max-rounds. Pass = no deadlock and last-frame period <= the network's slowest MILP node * (1 + slack).
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

MAX_TRIES = 10                     # the extended _SIZING_SCHEDULE has 10 entries (up/int/fnl default to 6)
FRAMES = 3
INTRA_GROWTH = 4                  # a full intra-block FIFO of a deadlocked chain is grown x4 per round (every round of the real nets costs minutes); interfaces x2


# ---------------------------------------------------------------------------------------------- MILP entry -> explicit folds
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
        f["Bias"] = 1                                       # no MILP node; the real build (partition 7) uses PE 1
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
    F = max([v["cycles"] for v in per_layer.values() if v["stage"] == stage] + [v["cycles"] for v in extra.values() if v["stage"] == stage])
    return f, int(F), warn


def build_blocks(d: dict, a, ctx) -> tuple[list, list[str]]:
    """[(stage, kind, BottleneckResult built from the MILP folds)], warnings. Asserts that the model landed on exactly the MILP's pe / simd / thr_pe."""
    geoms, extras, geom, xnode, dmap, kinds = ctx
    per_layer, extra = d["per_layer"], d["extra_nodes"]
    blocks, warns, prev = [], [], None
    for stage in net_fold.block_order(geoms):
        kind = net_fold.block_kind(stage)
        folds, F, w = explicit_for_block(stage, kind, per_layer, extra, prev)
        warns += w
        if kind == "reg":                                   # model_bottleneck takes an integer cycles-per-pixel target
            px = geom[f"{stage}.reduce.0"].hin * geom[f"{stage}.reduce.0"].win
            F = -(-F // px) * px
        r, lf, xf, _slowest, _fifos, _fw = net_fold.run_block(stage, geom, a.bits, F, prev_output=prev, explicit=folds)
        for name, (pe, simd, thr_pe, _cyc) in lf.items():
            m = per_layer[name]
            assert (pe, simd) == (m["pe"], m["simd"]) and (thr_pe is None or thr_pe == m.get("thr_pe")), (name, (pe, simd, thr_pe), (m["pe"], m["simd"], m.get("thr_pe")))
        for name, (pe, simd, _rs) in xf.items():
            if name in extra and name != f"{stage}.residual_add":
                assert pe == extra[name]["pe"], (name, pe, extra[name]["pe"])
        blocks.append((stage, kind, r))
        prev = block_output_name(stage, kind)
    return blocks, warns


# ---------------------------------------------------------------------------------------------- per-block simulation (parallel)
def _verify_keep(args):
    """verify_with_sim that keeps the result of a failing block (verify_with_sim sets r.fifo_graph / r.verification before it raises)."""
    kind, r, stage, F_global = args
    import bottleneck, dn_bottleneck, fnl_block, int_bottleneck, up_bottleneck
    ver = {"reg": bottleneck, "dn": dn_bottleneck, "up": up_bottleneck, "init": int_bottleneck, "final": fnl_block}[kind].verify_with_sim
    try:
        ver(r, max_tries=MAX_TRIES)
        return r, None
    except RuntimeError as e:
        err = str(e)
    v = r.verification
    if v.get("deadlock"):
        return r, err
    # The block cannot reach its own cycle budget at ANY depth (the sliding window serves one frame at a time: a window-fill gap per frame that no FIFO hides, large for the dilated
    # convs): size it for the rate it does reach (the first schedule entry that gets there, then shrunk) and accept it when that still meets the network-wide frame budget.
    T = r.params["T"]
    achieved = v["steady_cyc_px"]
    period = achieved * r.params["hout"] * r.params["wout"]
    try:
        ver(r, max_tries=MAX_TRIES, tol=achieved / T - 1 + 0.005, tol_soft=achieved / T - 1 + 0.005)
    except RuntimeError as e2:
        return r, f"{err}; relaxed to the achieved rate: {e2}"
    r.verification["rate_miss"] = dict(T=T, achieved_cyc_px=achieved, over_pct=round(100 * (achieved / T - 1), 2), period_cycles=int(period), limit=int(F_global), accepted=bool(period <= F_global))
    return r, (None if period <= F_global else f"{err}; and its achieved period {period:.0f} cycles/frame exceeds the network-wide budget {F_global}")


def verify_blocks_keep(blocks: list, workers: int, F_global: int) -> tuple[list, dict]:
    key = lambda kind, r: repr((kind, sorted(r.params.items()), [(n.name, n.pe, n.simd) for n in r.nodes]))
    todo: dict = {}
    for stage, kind, r in blocks:
        todo.setdefault(key(kind, r), (kind, r, stage, F_global))
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


def run_chain_rounds(blocks: list, F_net: int, slack: float, max_rounds: int, log=print, F_global: int | None = None) -> dict:
    """Chain the blocks (inter-block depth 2) and grow FIFOs until the chain neither deadlocks nor misses its limit: the network-wide frame budget F_global when given
    (the throughput requirement), else F_net * (1 + slack). `meets_milp_node` records separately whether the period is within F_net * (1 + slack) of the MILP's own slowest node. Mutates the blocks' fifo_graph."""
    n = len(blocks)
    depths = [net_fifo.MIN_DEPTH] * (n - 1)
    rounds, grown = [], {}                          # grown: (stage, fifo name) | ("iface", i) -> [first depth, last depth]
    limit = F_global if F_global else F_net * (1 + slack)
    milp_limit = F_net * (1 + slack)
    ok = False
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
        ok = (not res["deadlock"]) and res["period"] <= limit

        def label(f):
            if id(f) in iface:
                return f"iface{iface[id(f)]} ({blocks[iface[id(f)]][0]} -> {blocks[iface[id(f)] + 1][0]})"
            i, name = owner.get(id(f), (None, f.name))
            return f"{blocks[i][0]}:{name}" if i is not None else f.name
        info = dict(round=rnd, meets_milp_node=bool((not res["deadlock"]) and res["period"] <= milp_limit), deadlock=res["deadlock"], period=(None if res["period"] == float("inf") else int(res["period"])), first_out=int(res["first_out"]), ok=ok,
                    full=[label(f) for f in res["full"]], seconds=round(time.time() - t0))
        rounds.append(info)
        log(f"  chain round {rnd}: deadlock={res['deadlock']} period={info['period']} (limit {limit:.0f}) first_out={info['first_out']} full={len(info['full'])} [{info['seconds']} s]", flush=True)
        if ok:
            break
        targets = res["full"] if res["deadlock"] else [f for f in res["saturated"] if id(f) in iface]    # a rate miss without deadlock: only the interfaces can help
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
    return dict(ok=ok, rounds=rounds, interface_depths=depths, grown=[dict(where=k[0], fifo=(k[1] if k[0] != "iface" else f"iface{k[1]}"), depth_before=v[0], depth_after=v[1]) for k, v in grown.items()],
                last=rounds[-1] if rounds else None, target=F_net, limit=limit, milp_limit=milp_limit)


# ---------------------------------------------------------------------------------------------- main / output
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folding_json", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="output file tag: layer_bits_folding_<tag>.json")
    ap.add_argument("--config", default="config_12_dense_relu_nearest_upsample_256")
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--slack", type=float, default=0.03, help="accepted last-frame period overshoot of the whole-net run over the slowest MILP node")
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--reuse-blocks", action="store_true", help="reuse <out-dir>/blocks.pkl (the simulated blocks of an earlier run) instead of re-simulating every block")
    ap.add_argument("--no-chain", action="store_true", help="per-block simulation only (skip the whole-net runs)")
    a = ap.parse_args()
    raise SystemExit(run(a))


def run(a) -> int:
    from bottleneck import WWIDTH_MAX  # noqa: F401  (explicit folds skip the width cap)
    d = json.loads(a.folding_json.read_text())
    assert d["status"] == "Optimal", d["status"]
    finn_milp.load_config(a.config)
    finn_milp.CANDIDATE_BITS = tuple(sorted(set(finn_milp.CANDIDATE_BITS) | {a.bits}))
    model, geoms, extras, _pred, dmap, kinds = finn_milp.build_model_and_graph()
    ctx = (geoms, extras, {g.name: g for g in geoms}, {n.geom.name: n for n in extras}, dmap, kinds)
    t0 = time.time()
    blocks, warns = build_blocks(d, a, ctx)
    print(f"{a.tag}: {len(blocks)} blocks built from the MILP folding ({len(warns)} warnings)", flush=True)
    F_global = int(d["_diagnostics"]["max_node_cycles"])
    cache = a.out_dir / "blocks.pkl"
    if a.reuse_blocks and cache.exists():
        blocks, errors = pickle.loads(cache.read_bytes())
        print(f"reusing the simulated blocks of {cache}", flush=True)
    else:
        blocks, errors = verify_blocks_keep(blocks, a.workers, F_global)
        a.out_dir.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps((blocks, errors)))
    for stage, err in errors.items():
        print(f"  BLOCK FAILS ALONE: {stage}: {err}", flush=True)
    nodes = {**d["per_layer"], **d["extra_nodes"]}
    F_net = int(max([v["cycles"] for v in nodes.values()] + [r.verification["rate_miss"]["period_cycles"] for _, _, r in blocks if "rate_miss" in r.verification]))
    chain = None
    if not a.no_chain:
        print(f"whole-net chain: pass = no deadlock and period <= {F_global} cycles/frame (network budget); slowest MILP node / block {F_net}, inter-block depth 2, up to {a.max_rounds} rounds", flush=True)
        chain = run_chain_rounds(blocks, F_net, a.slack, a.max_rounds, F_global=F_global)
    out = assemble_output(d, blocks, chain, errors, warns, a, F_net, time.time() - t0)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    path = a.out_dir / f"layer_bits_folding_{a.tag}.json"
    path.write_text(json.dumps(out, indent=1))
    rep = out["fifo_sim"]
    print(f"{a.tag}: wrote {path}\n  blocks failing alone: {len(errors)}, chain ok: {rep['chain_ok']}, intra FIFO BRAM18 {rep['bram18_milp_estimate']} (MILP) -> {rep['bram18_sim']} (sim)", flush=True)
    return 0 if (not errors and (chain is None or chain["ok"])) else 2


def assemble_output(d: dict, blocks: list, chain: dict | None, errors: dict, warns: list[str], a, F_net: int, seconds: float) -> dict:
    out = copy.deepcopy(d)
    g = out["_diagnostics"]
    milp_depth = {(f["stage"], f["producer"], f["consumer"]): f["depth"] for f in d.get("intra_block_fifos", [])}
    fm = g.pop("fifo_model", None)
    fm_tot = (fm or {}).get("totals", {})
    inter_old = d.get("inter_block_fifos", [])
    est_lut = fm_tot.get("lut", 0) + fm_tot.get("dwc_lut", 0) + sum(f.get("lut", 0) for f in inter_old)
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
        source=str(a.folding_json.name), tag=a.tag, bits=a.bits, seconds=round(seconds), net_target_cycles=F_net, slack=a.slack, workers=a.workers,
        blocks={stage: dict(kind=kind, ok=(stage not in errors), error=errors.get(stage), tries=r.verification.get("tries"), deadlock=r.verification.get("deadlock"), T=r.params.get("T"),
                            steady_cyc_px=r.verification.get("steady_cyc_px"), rate_miss=r.verification.get("rate_miss"), uniform_depth=r.verification.get("uniform_depth"), elastic_depth=r.verification.get("elastic_depth"),
                            full_fifos=(r.verification.get("full_fifos") or []) if stage in errors else [], warnings=list(r.warnings))
                for stage, kind, r in blocks},
        chain=chain, chain_ok=(None if chain is None else chain["ok"]), build_warnings=warns,
        bram18_milp_estimate=est_bram, bram18_sim=g["intra_block_fifo_bram18"] + g["inter_block_fifo_bram18"],
        note="folding identical to the source MILP json; FIFO depths = the smallest per-block depths that sustain each block's own slowest MILP node in simulation, grown further by the whole-net chain on a deadlock",
    )
    return out


def net_fold_min() -> int:
    return net_fifo.MIN_DEPTH


if __name__ == "__main__":
    main()
