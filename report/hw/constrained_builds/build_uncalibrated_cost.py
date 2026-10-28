"""UNCALIBRATED = STOCK-FINN resource estimate of the four constrained S12-512 designs (ILP/artifacts/S12_512_ablation_v2): the SAME final designs as modelled_vs_real_cost.csv (the ILP folding after the
FIFO simulation: per-layer PE / SIMD / bits, extra nodes, simulated FIFO depths, DWCs), but every node priced by FINN v0.10.1's OWN estimators (`lut_estimation()`, `bram_estimation()`,
`uram_estimation()`, `dsp_estimation()`), called from the checked-out FINN source through stock_finn_cost.py. No constant of this repo's cost model enters the uncalibrated column.
Output: uncalibrated_cost.json and uncalibrated_cost_breakdown.csv; plot_constrained_builds.py merges the totals into modelled_vs_real_cost.csv (model_uncal_* columns).

What stock FINN says per node (see stock_finn_cost.py for the call details):
  MVAU_rtl   0 LUT, DSP = ceil(PE/4) * SIMD, weight BRAM by the FINN-R formula (ram_style "auto" is estimated like "block")     VVAU_hls   FINN-R MVU formula, DSP = PE * ceil((W+A)/48)
  SWG_rtl    300 + line-buffer LUTRAM on the padded feature map                       Thresholding_rtl   LUTRAM / BRAM / URAM of get_memory_estimate(); with the bridge's depth_trigger_bram and
  DWC        its shift-based formula                                                  depth_trigger_uram = 0 the estimator puts every threshold stage in URAM (stock quirk, reproduced)
  StreamingFIFO_rtl   per impl_style (rtl <= 256 words, Vivado block RAM above)       ChannelwiseOp_hls (final bias)   comparator + LUTRAM cost
  AddStreams, DuplicateStreams, StreamingConcat, Upsample, LabelSelect, StreamingMaxPool, Pool, FMPadding   no estimator in v0.10.1: 0
The graph STRUCTURE (which nodes exist, their PE / SIMD / bit widths / geometry, the FIFO depths and the DWC set) is the modelled design's; only the pricing differs. The accumulator width used for
thresholds is the model's own FINN-R bound.

The script first re-prices every node with the CURRENT calibrated model and checks the totals against the stored `_diagnostics` (plumbing check, must match), then prices with stock FINN.
Runs in the MILP container (needs torch for the geometry trace and qonnx for the FINN datatypes):
    docker exec lightmunet_dev python3 report/MILP/constrained_builds/build_uncalibrated_cost.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for p in (REPO_ROOT / "enet", REPO_ROOT / "ILP", REPO_ROOT / "ILP" / "utils", REPO_ROOT / "ILP" / "analytical", HERE):
    sys.path.insert(0, str(p))

import bottleneck  # noqa: E402
import finn_cost_model as fcm  # noqa: E402
import finn_milp as fm  # noqa: E402
import stock_finn_cost as stock  # noqa: E402

ARTIFACTS = REPO_ROOT / "ILP" / "artifacts" / "S12_512_ablation_v2"
# scheme -> (solve folder, simfifo json); same table as plot_constrained_builds.SCHEMES
SCHEMES = {
    "int4": ("cap40_fps150/uniform_int4", "layer_bits_folding_int4_cap40_simfifo.json"),
    "int6": ("cap40_fps150/uniform_int6", "layer_bits_folding_int6_cap40_simfifo.json"),
    "int8": ("relaxed_fps150/uniform_int8", "layer_bits_folding_int8_fps150_simfifo.json"),
    "mixed468": ("cap40_fps150/mixed_468", "layer_bits_folding_mixed468_cap40_simfifo.json"),
}
KEYS = ("LUT", "BRAM", "DSP", "URAM")


class Pricer:
    """Prices one stored design node by node: `uncal=False` with the calibrated model (reproduces the stored totals), `uncal=True` with FINN's own estimators. Fills a per-kind breakdown."""

    def __init__(self, d: dict, geom: dict, enodes: dict, uncal: bool):
        self.d, self.geom, self.enodes, self.uncal = d, geom, enodes, uncal
        self.by_kind = defaultdict(lambda: dict.fromkeys(KEYS, 0.0))
        self.total = dict.fromkeys(KEYS, 0.0)
        self.max_err = 0.0

    def add(self, kind: str, lut: float = 0.0, bram: float = 0.0, dsp: float = 0.0, uram: float = 0.0) -> None:
        for key, value in zip(KEYS, (lut, bram, dsp, uram)):
            self.by_kind[kind][key] += value
            self.total[key] += value

    def add_stock(self, kind: str, est: dict) -> None:
        self.add(kind, est["LUT"], est["BRAM"], est["DSP"], est["URAM"])

    def conv(self, kind: str, g, w: int, a: int, pe: int, simd: int, cost: dict, kw: dict, thr_ram_style: str, swu_ram_style: str, vvau: bool) -> float:
        """One MVAU / VVAU with its SWG and standalone threshold. Returns the calibrated LUT (for the per-node check). Components go to the breakdown as `kind:component`."""
        if self.uncal:
            hpad, wpad = g.hin + 2 * g.ph, g.win + 2 * g.pw         # FMPadding is a separate FINN node: the SWG sees the padded map
            self.add_stock(kind + ":swg", stock.sliding_window_rtl(cost["simd_swu"], a, g.cin, g.kh, g.kw, hpad, wpad, g.sh, g.sw, g.dh, g.dw,
                                                                 simd > 1 if vvau else simd > g.cin, vvau, swu_ram_style))
            if vvau:
                self.add_stock(kind + ":mvu", stock.vvau_hls(pe, simd, w, a, g.cout, g.kh, g.kw))
            else:
                self.add_stock(kind + ":mvu", stock.mvau_rtl(pe, simd, w, g.cin * g.kh * g.kw // g.groups, g.cout))
            if kw.get("no_activation", True):
                self.add_stock(kind + ":thr", stock.thresholding_rtl(cost["thr_pe"], g.cout, cost["acc_bits"], a, thr_ram_style))
        else:
            self.add(kind + ":swg", cost["swu_lut"], cost["swu_bram18"], 0)
            self.add(kind + ":mvu", cost["mvu_lut"], cost["wm_bram18"], cost["total_dsp"])
            self.add(kind + ":thr", cost["thr_lut"], cost.get("thr_bram18", 0), 0)
        return cost["total_lut"]

    def run(self) -> dict:
        d = self.d
        for name, e in d["per_layer"].items():
            if e.get("frozen"):      # the frozen depthwise conv is MIRRORED into per_layer for the FINN bridge; its resources are counted once, as an extra node
                continue
            g, w, a = self.geom[name], e.get("weight_bits"), e["act_bits"]
            if g.op_type == "MaxPool2d":
                if e["variant"] == fm.VARIANT_POOL_SWG:      # InferPool route: depthwise SWG + Pool_hls (no estimator -> 0)
                    cost = fcm.pool_swg_cost(g, a, e["pe"])
                    if self.uncal:
                        self.add_stock("maxpool", stock.sliding_window_rtl(e["pe"], a, g.cin, g.kh, g.kw, g.hin, g.win, g.sh, g.sw, g.dh, g.dw, False, True, "distributed"))
                    else:
                        self.add("maxpool", cost["total_lut"], cost["swu_bram18"], 0)
                else:                                          # StreamingMaxPool_hls: no estimator -> 0
                    cost = fcm.maxpool_cost(g, a)
                    if self.uncal:
                        self.add_stock("maxpool", dict(stock.ZERO))
                    else:
                        self.add("maxpool", cost["total_lut"], cost["swu_bram18"], 0)
                lut = cost["total_lut"]
            else:
                kw = fm._variant_cost_kwargs(e["variant"], e["force_dsp"])
                cost = fcm.layer_cost_pe_simd(g, w, a, e["pe"], e["simd"], fcm.RAM_STYLE_AUTO, swu_ram_style=fm._variant_swu_style(e["variant"]), thr_ram_style=e["thr_ram_style"], **kw)
                lut = self.conv("depthwise" if g.groups > 1 else "mvau", g, w, a, e["pe"], e["simd"], cost, kw, e["thr_ram_style"], fm._variant_swu_style(e["variant"]), g.groups > 1)
            if not self.uncal:
                self.max_err = max(self.max_err, abs(lut - e["lut_calibrated"]))
        rtl_kw = fm._variant_cost_kwargs(fm.VARIANT_RTL_DSP_NOACT1, True)
        final_acc_bits = d["per_layer"]["final"]["acc_bits"]
        for name, e in d["extra_nodes"].items():
            kind, g, pe = e["kind"], self.enodes[name].geom, e["pe"]
            if kind in fm.THRESHOLD_KINDS:
                bits = e["act_bits"]
                in_bits = bits + 2 if kind in fm.TIED_BIT_KINDS else fcm.THR_DEFAULT_IN_BITS
                cost = fcm.threshold_node_cost(g, bits, pe, ram_style=e["ram_style"], in_bits=in_bits)
                if self.uncal:
                    self.add_stock(kind, stock.thresholding_rtl(pe, g.cout, in_bits, bits, e["ram_style"]))
                else:
                    self.add(kind, cost["total_lut"], cost["thr_bram18"], 0)
                lut = cost["total_lut"]
            elif kind in fcm.STREAM_NODE_KINDS:
                cost = fcm.stream_node_cost(kind, g, pe)
                if self.uncal:
                    self.add_stock(kind, stock.channelwise_bias(pe, g.cout, final_acc_bits) if kind == "bias" else dict(stock.ZERO))
                else:
                    self.add(kind, cost["total_lut"])
                lut = cost["total_lut"]
            elif kind == "pad_mvau":
                w, a = fm.PAD_MVAU_BITS
                kw = {**rtl_kw, "no_activation": False}
                cost = fcm.layer_cost_pe_simd(g, w, a, pe, e["simd"], fcm.RAM_STYLE_AUTO, swu_ram_style="distributed", **kw)
                lut = self.conv("pad_mvau", g, w, a, pe, e["simd"], cost, kw, "block", "distributed", False)
            elif kind == fm.FROZEN_DW_KIND:
                a = e["act_bits"]
                kw = {**rtl_kw, "no_activation": True}
                cost = fcm.layer_cost_pe_simd(g, fm.FROZEN_DW_WEIGHT_BITS, a, pe, e["simd"], fcm.RAM_STYLE_AUTO, swu_ram_style="distributed", thr_ram_style="block", **kw)
                lut = self.conv("frozen_dw", g, fm.FROZEN_DW_WEIGHT_BITS, a, pe, e["simd"], cost, kw, "block", "distributed", True)
            else:
                raise ValueError(f"unknown extra kind {kind}")
            if not self.uncal:
                self.max_err = max(self.max_err, abs(lut - e["lut_calibrated"]))
        # the simulation's own DWC set (block_verification), NOT the MILP's pre-simulation `dwcs` list
        for blk in d["block_verification"].values():
            for dw in blk["dwcs"]:
                if self.uncal:
                    self.add_stock("dwc", stock.dwc(dw["in_width"], dw["out_width"]))
                else:
                    self.add("dwc", dw["lut"])
        for f in d["intra_block_fifos"]:
            if self.uncal:
                self.add_stock("fifo_intra", stock.streaming_fifo(f["width_bits"], f["depth"]))
            else:       # the list's mem_lut is rounded per FIFO; the model's own pricing is exact
                m = bottleneck.fifo_memory(f["width_bits"], f["depth"], "auto")
                self.add("fifo_intra", m["lut"], m["bram18"])
        for f in d["inter_block_fifos"]:
            if self.uncal:
                self.add_stock("fifo_inter", stock.streaming_fifo(f["width_bits"], f["depth"]))
            else:
                self.add("fifo_inter", f["lut"], f["bram18"])
        return self.total


def main() -> int:
    out, rows, loaded = {}, [], None
    for key, (solve, sim_json) in SCHEMES.items():
        run_args = json.loads((ARTIFACTS / solve / "run_args.json").read_text())["shared_args"]
        if loaded != run_args["config"]:
            fm.load_config(run_args["config"])
            _, geometries, extra_nodes, *_ = fm.build_model_and_graph()
            geom, enodes, loaded = {g.name: g for g in geometries}, {n.geom.name: n for n in extra_nodes}, run_args["config"]
        d = json.loads((ARTIFACTS / solve / "simfifo" / sim_json).read_text())
        dg = d["_diagnostics"]
        ref = (dg["total_lut_with_all_fifos"], dg["total_bram18k_with_all_fifos"] + dg.get("total_uram18", 0.0), float(dg["total_dsp"]))
        cal = Pricer(d, geom, enodes, uncal=False)
        c = cal.run()
        if abs(c["LUT"] - ref[0]) > 1.0 or abs(c["BRAM"] - ref[1]) > 0.5 or abs(c["DSP"] - ref[2]) > 0.5:
            raise SystemExit(f"{key}: calibrated re-pricing {c['LUT']:.1f} LUT / {c['BRAM']:.1f} BRAM / {c['DSP']:.0f} DSP does not reproduce the stored {ref[0]:.1f} / {ref[1]:.1f} / {ref[2]:.0f} "
                             f"(max node error {cal.max_err:.3g}) -- fix the plumbing before trusting the stock numbers")
        unc = Pricer(d, geom, enodes, uncal=True)
        u = unc.run()
        out[key] = {"config": run_args["config"], "calibrated": {k.lower() if k != "BRAM" else "bram18": c[k] for k in KEYS}, "uncalibrated": {k.lower() if k != "BRAM" else "bram18": u[k] for k in KEYS}}
        print(f"{key:9s} calibrated {c['LUT']:9.0f} LUT {c['BRAM']:6.0f} BRAM18 {c['DSP']:5.0f} DSP (stored {ref[0]:.0f} / {ref[1]:.0f} / {ref[2]:.0f}, max node err {cal.max_err:.2g})  |  "
              f"stock FINN {u['LUT']:9.0f} LUT {u['BRAM']:6.0f} BRAM18 {u['DSP']:5.0f} DSP {u['URAM']:5.0f} URAM")
        for kd in sorted(set(cal.by_kind) | set(unc.by_kind)):
            cc, uu = cal.by_kind.get(kd, dict.fromkeys(KEYS, 0.0)), unc.by_kind.get(kd, dict.fromkeys(KEYS, 0.0))
            rows.append([key, kd] + [f"{x:.0f}" for k in KEYS for x in (cc[k], uu[k])])
    (HERE / "uncalibrated_cost.json").write_text(json.dumps(out, indent=2))
    with open(HERE / "uncalibrated_cost_breakdown.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "kind"] + [f"{k.lower() if k != 'BRAM' else 'bram18'}_{v}" for k in KEYS for v in ("calibrated", "stock_finn")])
        w.writerows(rows)
    print(f"Wrote {HERE / 'uncalibrated_cost.json'} and uncalibrated_cost_breakdown.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
