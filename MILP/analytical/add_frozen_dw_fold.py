"""Fold the bilinear decoder's FROZEN depthwise 3x3 conv AFTER the MILP solve, outside the MILP, and add its cost to the arm's totals.

    python3 MILP/analytical/add_frozen_dw_fold.py <layer_bits_folding_ARM.json> --config <config> [--simfifo <layer_bits_folding_ARM_simfifo.json>] [--out <path>]

LayerQuantEnetFINN's bilinear decoder puts a frozen INT8 depthwise 3x3 conv (`<stage>.main_up.1`: nearest upsample -> depthwise conv -> the block's skip_quant threshold) in the main branch of every up block. It has no
quantizer site and no bit choice, so the MILP graph (and with it the folding json the FINN bridge reads) has no entry for it; left to FINN's default fold (PE = SIMD = 1) it would need 9 * C * H*W ~ 2.36M cycles
per frame (up4: 16 ch at 128x128, up5: 4 ch at 256x256) -- 6x the 400k-cycle budget. This script gives it a fold:

  * the geometry is net_fold.add_bilinear_dw's (cin = cout = groups = C, 3x3, pad 1, at the upsampled resolution), weights 8 bit, activations at the block's shared quantizer (`<stage>.residual_add`, the
    stream the nearest upsampler carries);
  * its options are finn_milp.candidate_folds (PE | C, SIMD | 9), priced exactly like the MILP prices a layer (layer_cost_pe_simd, no fused activation: the threshold is the existing skip_quant node,
    calibrated_lut / calibrated_bram18k), plus the FIFOs the MILP's FIFO model would add for a padded 3x3 conv inside an up block (the prefetch FIFO of the next frame's first window, the 32-word FMPadding -> SWG ->
    VVAU -> threshold FIFOs; width converters to the neighbouring nodes are NOT priced);
  * it picks the option with the least resources (the MILP's --min-resources objective: the equal-weight mean of the LUT / BRAM18K / DSP fractions of the board) whose cycles per frame are within the MILP's per-node
    budget (_diagnostics.max_node_cycles).

Writes `<folding json stem>_dw.json` next to the input: the original plus per_layer['<stage>.main_up.1'] (pe, simd, cycles, resources, frozen = True) and _diagnostics['bilinear_dw'] (the totals before / after
and the cap check; with --simfifo also the post-simulation totals). The original json is left untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "enet", REPO / "MILP", REPO / "MILP" / "utils", REPO / "MILP" / "analytical"):
    sys.path.insert(0, str(p))

import finn_milp  # noqa: E402
import net_fold  # noqa: E402
from bottleneck import BLOCK_FIFO_DEPTH, fifo_memory  # noqa: E402
from finn_cost_model import RAM_STYLE_AUTO, calibrated_bram18k, calibrated_lut, layer_cost_pe_simd  # noqa: E402
from fifo_model import n_fill_px  # noqa: E402

FROZEN_WEIGHT_BITS = 8
XCZU7EV = finn_milp.XCZU7EV


def frozen_dw_geometries(geoms: list, dmap: dict) -> dict:
    """{layer name: LayerGeometry} of the frozen depthwise convs (net_fold.add_bilinear_dw on copies, so the caller's graph is untouched)."""
    g2, d2 = list(geoms), {k: list(v) for k, v in dmap.items()}
    names = net_fold.add_bilinear_dw(g2, d2)
    by = {g.name: g for g in g2}
    return {n: by[n] for n in names}


def _option_cost(layer, a_bits: int, pe: int, simd: int, ram_style: str, variant: str, force_dsp: bool) -> dict:
    """One option's resources, priced like finn_milp prices a layer, plus the FIFOs fifo_model adds inside an up block."""
    kw = finn_milp._variant_cost_kwargs(variant, force_dsp)
    cost = layer_cost_pe_simd(layer, FROZEN_WEIGHT_BITS, a_bits, pe, simd, RAM_STYLE_AUTO, swu_ram_style="distributed", thr_ram_style=ram_style, **kw)
    lut = calibrated_lut(cost["total_lut"], FROZEN_WEIGHT_BITS, a_bits, force_dsp=finn_milp._calibration_force_dsp(kw))
    bram = calibrated_bram18k(cost["swu_bram18"] + cost["wm_bram18"] + cost.get("thr_bram18", 0), FROZEN_WEIGHT_BITS, a_bits, force_dsp=kw["force_dsp"])
    # FIFOs of a padded 3x3 conv (fifo_model._build_constants / _build_edge_fifos): prefetch in front of FMPadding, then FMPadding -> SWG -> VVAU -> threshold at the up block's depth
    simd_swu = cost["simd_swu"]
    parallel = simd > layer.cin // layer.groups if layer.groups > 1 else simd > layer.cin
    prefetch_words = n_fill_px(layer) * math.ceil(layer.cin / simd_swu) + 2
    fifos = [("prefetch", simd_swu * a_bits, prefetch_words), ("fmpad->swg", simd_swu * a_bits, BLOCK_FIFO_DEPTH),
             ("swg->mvau", (layer.kh * layer.kw if parallel else 1) * simd_swu * a_bits, BLOCK_FIFO_DEPTH), ("mvau->thr", pe * cost["acc_bits"], BLOCK_FIFO_DEPTH)]
    f_lut = f_bram = 0.0
    for _, width, depth in fifos:
        m = fifo_memory(int(width), int(depth))
        f_lut += m["lut"]
        f_bram += m.get("bram18", 0)
    return dict(cost=cost, lut=lut, bram=bram, dsp=cost["total_dsp"], fifo_lut=f_lut, fifo_bram=f_bram, prefetch_words=prefetch_words)


def fold_frozen_dw(layer, a_bits: int, budget_cycles: float, force_dsp: bool = True) -> dict:
    """The least-resource (PE, SIMD) of the frozen depthwise conv whose cycles per frame are within budget_cycles. Raises ValueError when none is."""
    best = None
    for pe, simd, ram_style, variant in finn_milp.candidate_folds(layer):
        o = _option_cost(layer, a_bits, pe, simd, ram_style, variant, force_dsp)
        if o["cost"]["cycles"] > budget_cycles:
            continue
        lut, bram, dsp = o["lut"] + o["fifo_lut"], o["bram"] + o["fifo_bram"], o["dsp"]
        score = (lut / XCZU7EV["LUT"] + bram / XCZU7EV["BRAM_18K"] + dsp / XCZU7EV["DSP"]) / 3
        key = (score, pe * simd)
        if best is None or key < best[0]:
            best = (key, pe, simd, ram_style, variant, o)
    if best is None:
        raise ValueError(f"{layer.name}: no (PE, SIMD) reaches {budget_cycles} cycles/frame")
    _, pe, simd, ram_style, variant, o = best
    c = o["cost"]
    # FINN's depthwise invariant (also hardware/ bridge scripts): FMPadding.SIMD == ConvolutionInputGenerator.SIMD == VVAU.PE (channel parallelism); the VVAU's own SIMD is the kernel parallelism (| 9,
    # > 1 = the SWG's parallel_window). cost["simd_swu"] is the SWG / FMPadding SIMD the cost model prices a depthwise conv with.
    assert c["simd_swu"] == pe, (layer.name, "SWG / FMPadding SIMD must equal the VVAU PE", c["simd_swu"], pe)
    assert layer.cout % pe == 0 and 9 % simd == 0, (layer.name, pe, simd)
    entry = dict(c)
    entry.update(stage=layer.stage, pe=pe, simd=simd, thr_ram_style=ram_style, variant=variant, force_dsp=force_dsp, weight_bits=FROZEN_WEIGHT_BITS, act_bits=a_bits,
                 lut_calibrated=o["lut"], bram18k_calibrated=o["bram"], fifo_lut=o["fifo_lut"], fifo_bram18=o["fifo_bram"], prefetch_words=o["prefetch_words"], frozen=True,
                 swg_simd=c["simd_swu"], fmpad_simd=c["simd_swu"], parallel_window=bool(simd > 1), invariant="fmpadding.simd == swg.simd == vvau.pe")
    return entry


def run(folding_json: Path, config: str, simfifo: Path | None, out: Path | None) -> dict:
    d = json.loads(folding_json.read_text())
    assert d["status"] == "Optimal", d["status"]
    if any(n.endswith(finn_milp.FROZEN_DW_SUFFIX) for n in d["per_layer"]):
        raise SystemExit(f"{folding_json}: the MILP already folded the frozen depthwise conv (extra node `{finn_milp.FROZEN_DW_KIND}`, finn_milp.py since 2026-10-13); nothing to add. "
                         "This script is for folding jsons from solves made before that.")
    finn_milp.load_config(config)
    finn_milp.CANDIDATE_BITS = tuple(sorted(set(finn_milp.CANDIDATE_BITS) | {FROZEN_WEIGHT_BITS}))
    _model, geoms, _extras, _pred, dmap, _kinds = finn_milp.build_model_and_graph()
    sites = net_fold.sites_for(config, folding_json)
    diag = d["_diagnostics"]
    budget = int(diag["max_node_cycles"])
    force_dsp = bool(diag.get("force_dsp", True))
    entries = {}
    for name, layer in frozen_dw_geometries(geoms, dmap).items():
        a_bits = int(sites["layer_act_bits"][f"{layer.stage}.residual_add"])        # the main branch's stream is quantized by the add's shared quantizer
        entries[name] = fold_frozen_dw(layer, a_bits, budget, force_dsp)
        # the threshold behind the depthwise VVAU IS the block's skip_quant node. The FINN bridge applies an entry's thr_pe / thr_ram_style to the threshold that follows its VVAU and (the VVAU being a claimed
        # layer) no longer treats that threshold as a join skip_quant, so the entry must carry skip_quant's own PE and memory style
        sq = d["extra_nodes"][f"{layer.stage}.skip_quant"]
        entries[name].update(thr_pe=int(sq["pe"]), thr_ram_style=sq["ram_style"], mvau_noAct=True)
    add = dict(lut=sum(e["lut_calibrated"] + e["fifo_lut"] for e in entries.values()), bram18=sum(e["bram18k_calibrated"] + e["fifo_bram18"] for e in entries.values()),
               dsp=sum(e["total_dsp"] for e in entries.values()))
    pct = lambda v, key: 100 * v / XCZU7EV[key]
    milp = dict(lut=diag["total_lut_calibrated"], bram18=diag["total_bram18k_calibrated"], dsp=diag["total_dsp"])
    cap = {"lut": diag["max_lut_fraction"], "bram18": diag["max_bram_fraction"], "dsp": diag["max_dsp_fraction"]}
    ra = folding_json.with_name("run_args.json")
    if ra.exists():                                    # the caps the arm was SOLVED under (a --lexicographic run records pass 1's own use as the roof in _diagnostics)
        sa = json.loads(ra.read_text())["shared_args"]
        cap = {"lut": sa["max-lut-fraction"], "bram18": sa["max-bram-fraction"], "dsp": sa["max-dsp-fraction"]}
    rep = dict(note="frozen depthwise 3x3 conv of the bilinear decoder, folded after the solve (MILP/analytical/add_frozen_dw_fold.py); width converters to its neighbours are not priced",
               budget_cycles=budget, layers={n: dict(pe=e["pe"], simd=e["simd"], cycles=e["cycles"], act_bits=e["act_bits"], lut=e["lut_calibrated"] + e["fifo_lut"], bram18=e["bram18k_calibrated"] + e["fifo_bram18"],
                                                     dsp=e["total_dsp"]) for n, e in entries.items()},
               added=add, milp_before=milp, milp_with_dw={k: milp[k] + add[k] for k in milp}, cap_fraction=cap)
    rep["milp_with_dw_pct"] = dict(lut=pct(rep["milp_with_dw"]["lut"], "LUT"), bram18=pct(rep["milp_with_dw"]["bram18"], "BRAM_18K"), dsp=pct(rep["milp_with_dw"]["dsp"], "DSP"))
    rep["within_caps"] = dict(lut=rep["milp_with_dw_pct"]["lut"] <= 100 * cap["lut"] + 1e-9, bram18=rep["milp_with_dw_pct"]["bram18"] <= 100 * cap["bram18"] + 1e-9,
                              dsp=rep["milp_with_dw_pct"]["dsp"] <= 100 * cap["dsp"] + 1e-9)
    if simfifo is not None:
        s = json.loads(simfifo.read_text())["_diagnostics"]
        sim = dict(lut=s["total_lut_with_all_fifos"], bram18=s["total_bram18k_with_all_fifos"], dsp=s["total_dsp"])
        rep["sim_before"], rep["sim_with_dw"] = sim, {k: sim[k] + add[k] for k in sim}
        rep["sim_with_dw_pct"] = dict(lut=pct(rep["sim_with_dw"]["lut"], "LUT"), bram18=pct(rep["sim_with_dw"]["bram18"], "BRAM_18K"), dsp=pct(rep["sim_with_dw"]["dsp"], "DSP"))
    out_d = json.loads(json.dumps(d))
    out_d["per_layer"].update(entries)
    out_d["_diagnostics"]["bilinear_dw"] = rep
    out = out or folding_json.with_name(folding_json.stem + "_dw.json")
    out.write_text(json.dumps(out_d, indent=2))
    rep["out"] = str(out)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folding_json", type=Path)
    ap.add_argument("--config", required=True)
    ap.add_argument("--simfifo", type=Path, default=None, help="the same arm's net_explicit output: adds the post-simulation totals")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rep = run(a.folding_json, a.config, a.simfifo, a.out)
    for n, e in rep["layers"].items():
        print(f"  {n}: VVAU PE {e['pe']} SIMD {e['simd']}, SWG / FMPadding SIMD {e['pe']} (= PE) -> {e['cycles']} cycles (budget {rep['budget_cycles']}), act {e['act_bits']} b, +{e['lut']:.0f} LUT +{e['bram18']:.1f} BRAM18 +{e['dsp']} DSP")
    p = rep["milp_with_dw_pct"]
    print(f"  MILP with dw: LUT {p['lut']:.2f}% BRAM {p['bram18']:.2f}% DSP {p['dsp']:.2f}%   caps {rep['cap_fraction']}   within caps {rep['within_caps']}")
    print(f"wrote {rep['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
