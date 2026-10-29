"""Hardware cost of a FINN-style PReLU activation vs ReLU at uniform INT4 / INT6 / INT8, priced with the repo's FINN cost model (ILP/finn_cost_model.py). Assumptions: README.md.

Reference designs: ILP/artifacts/S12_dense_512_uniform_cap1_v2/uniform_int{4,6,8} (S12-dense bilinear 512, uniform bits, 250 FPS @ 100 MHz, MILP folding). In each, every ReLU activation site is a
standalone Thresholding_rtl (UINT<b>: 2**b-1 thresholds per channel): the 'reduce' / 'conv' / 'up' conv thresholds (PE = the conv's thr_pe, input = the accumulator, acc_bits) plus the post-residual
`out_act` and `initial.act` join thresholds (input INT10). The 82 sites are exactly the 82 QuantPReLU sites of LayerQuantEnetFINN(prelu=True).

Unclamped PReLU = two ReLUs in a fork/add:  PReLU(x) = ReLU(x) + alpha*min(x,0)  ->  DuplicateStreams -> [Thresholding(+ branch), Thresholding(- branch)] -> AddStreams.
The branches have disjoint support, so their sum is already on the signed b-bit output grid; the signed grid's thresholds are split over the two branches, i.e. two (b-1)-bit Thresholding nodes
instead of one b-bit node. The negative branch is assumed free (comparison direction / output sign folded into the node itself).
Optionally ("fusable") a site whose trained alphas all lie in (0, 1) is monotone and fuses into ONE b-bit threshold = ReLU cost; alphas come from the encoder-only checkpoint
(4_8_dense_dilation_alpha.csv); the 15 decoder sites have no trained alphas and count as not fusable.

    docker exec lightmunet_dev python3 /workspace/LightM-UNet/report/hw/activation/prelu_hw_cost.py [--bits 4 6 8]
Writes prelu_hw_cost_int<b>_sites.csv per width and prelu_hw_cost_summary.csv. Raw model units = calibrated (force_dsp factor 1.0). Not validated against a real PReLU build.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "ILP"))
import finn_cost_model as fcm  # noqa: E402

ARTIFACTS = REPO / "ILP/artifacts/S12_dense_512_uniform_cap1_v2"
ALPHA_CSV = HERE / "4_8_dense_dilation_alpha.csv"
DEVICE = {"lut": 230_400, "bram18": 624, "dsp": 1_728}   # xczu7ev


def site_to_node(module: str) -> str:
    """PReLU/QuantPReLU module name -> the cost-model node whose threshold it is."""
    for pat, rep in ((r"\.reduce\.2$", ".reduce.0"), (r"\.conv_bn_act\.2$", ".conv"), (r"\.conv\.2$", ".conv.0"), (r"\.up\.2$", ".up.0")):
        if re.search(pat, module):
            return re.sub(pat, rep, module)
    return module   # X.out_act, initial.act


def price(bits: int, alphas: dict[str, list[float]]) -> tuple[dict, list[dict]]:
    folder = ARTIFACTS / f"uniform_int{bits}"
    art = json.load(open(folder / f"layer_bits_folding_uniform_int{bits}.json"))
    shapes, per_layer, extra = art["dataflow_graph"]["shapes"], art["per_layer"], art["extra_nodes"]

    sites = []
    for name, p in per_layer.items():
        if re.search(r"\.(reduce\.0|conv|conv\.0|up\.0)$", name) and name != "initial.conv" and "thr_pe" in p:
            sites.append(dict(node=name, kind="conv_thr", pe=p["thr_pe"], in_bits=p["acc_bits"], style=p["thr_ram_style"], base_lut=p["thr_lut"]))
    for name, e in extra.items():
        if e["kind"] in ("out_act", "act"):
            sites.append(dict(node=name, kind=e["kind"], pe=e["pe"], in_bits=fcm.THR_DEFAULT_IN_BITS, style=e["ram_style"], base_lut=e["lut_calibrated"]))
    assert len(sites) == 82, f"expected 82 activation sites, found {len(sites)}"

    rows = []
    for s in sites:
        c, h, w = shapes[s["node"]]
        pe, style, ib = s["pe"], s["style"], s["in_bits"]
        relu_lut, relu_bram, _ = fcm._thresholding_rtl_cost(pe, bits, c, ram_style=style, in_bits=ib)
        br_lut, br_bram, _ = fcm._thresholding_rtl_cost(pe, bits - 1, c, ram_style=style, in_bits=ib)
        fork, add = pe * fcm._DUPSTREAMS_LUT_PER_PE, pe * fcm._ADDSTREAMS_LUT_PER_PE
        a = alphas.get(s["node"])
        rows.append(dict(
            site=s["node"], kind=s["kind"], channels=c, pixels=h * w, pe=pe, in_bits=ib, ram_style=style,
            relu_lut=relu_lut, relu_bram18=relu_bram, prelu_lut=fork + 2 * br_lut + add, prelu_bram18=2 * br_bram,
            fork_lut=fork, add_lut=add, recorded_relu_lut=s["base_lut"], fusable=int(bool(a) and min(a) > 0 and max(a) < 1),
        ))

    summ = next(csv.DictReader(open(folder / "summary.csv")))
    stale = sum(r["relu_lut"] - r["recorded_relu_lut"] for r in rows)   # artifact predates the 2026-10-17 threshold refit
    base_lut = float(summ["lut_pct_of_budget"]) / 100 * DEVICE["lut"] + stale
    base_bram = float(summ["bram_pct_of_budget"]) / 100 * DEVICE["bram18"]
    d_lut = sum(r["prelu_lut"] - r["relu_lut"] for r in rows)
    d_bram = sum(r["prelu_bram18"] - r["relu_bram18"] for r in rows)
    d_lut_fus = sum(r["prelu_lut"] - r["relu_lut"] for r in rows if not r["fusable"])
    d_bram_fus = sum(r["prelu_bram18"] - r["relu_bram18"] for r in rows if not r["fusable"])
    summary = dict(
        bits=bits, base_lut=round(base_lut), base_lut_pct_device=round(100 * base_lut / DEVICE["lut"], 1), base_bram18=round(base_bram), base_dsp=int(float(summ["total_dsp"])),
        relu_act_lut=round(sum(r["relu_lut"] for r in rows)), relu_act_bram18=round(sum(r["relu_bram18"] for r in rows)),
        prelu_act_lut=round(sum(r["prelu_lut"] for r in rows)), prelu_act_bram18=round(sum(r["prelu_bram18"] for r in rows)),
        fork_lut=round(sum(r["fork_lut"] for r in rows)), add_lut=round(sum(r["add_lut"] for r in rows)),
        delta_lut=round(d_lut), delta_lut_pct_design=round(100 * d_lut / base_lut, 1), delta_lut_pct_device=round(100 * d_lut / DEVICE["lut"], 1),
        delta_bram18=round(d_bram), delta_bram18_pct_device=round(100 * d_bram / DEVICE["bram18"], 2),
        n_fusable=sum(r["fusable"] for r in rows), delta_lut_if_fusable=round(d_lut_fus), delta_bram18_if_fusable=round(d_bram_fus),
        new_lut_pct_device=round(100 * (base_lut + d_lut) / DEVICE["lut"], 1), new_bram18_pct_device=round(100 * (base_bram + d_bram) / DEVICE["bram18"], 1),
    )
    return summary, rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8])
    args = p.parse_args()

    alphas: dict[str, list[float]] = {}
    with open(ALPHA_CSV) as f:
        for r in csv.DictReader(f):
            alphas.setdefault(site_to_node(r["module"]), []).append(float(r["alpha"]))

    summaries = []
    for bits in args.bits:
        s, rows = price(bits, alphas)
        summaries.append(s)
        with open(HERE / f"prelu_hw_cost_int{bits}_sites.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"INT{bits}: baseline {s['base_lut']:,} LUT ({s['base_lut_pct_device']}% of device) {s['base_bram18']} BRAM18 {s['base_dsp']} DSP | activation thresholds {s['relu_act_lut']:,} -> {s['prelu_act_lut']:,} LUT, "
              f"BRAM18 {s['relu_act_bram18']} -> {s['prelu_act_bram18']} | delta +{s['delta_lut']:,} LUT (+{s['delta_lut_pct_design']}% of design, +{s['delta_lut_pct_device']} pp of device; fork {s['fork_lut']:,} + add {s['add_lut']:,}) "
              f"{s['delta_bram18']:+d} BRAM18 | device LUT {s['base_lut_pct_device']}% -> {s['new_lut_pct_device']}% | if the {s['n_fusable']} fusable sites fuse: +{s['delta_lut_if_fusable']:,} LUT")
    with open(HERE / "prelu_hw_cost_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summaries[0]))
        wr.writeheader()
        wr.writerows(summaries)
    print(f"Wrote {HERE / 'prelu_hw_cost_summary.csv'}")


if __name__ == "__main__":
    main()
