"""Analytical model of the ENet FINAL deconvolution on FINN (companion of reg_/dn_/up_/int_bottleneck.py). NOT a bottleneck: a single path.

Reference: LayerQuantEnetFINN.final = qnn.QuantConvTranspose2d(c5, out_channels, kernel 2, stride 2, bias=True, bias_quant=Int32Bias,
weight Int8 quantizer) -- U4: 4 -> 5 channels, 128x128 -> 256x256. FINN lowers it (InferPixelPaddingDeconv) to zero insertion + a 2x2 window + MVAU;
the Int32 bias is an exact integer constant, so the Add after the MVAU becomes a ChannelwiseOp (no threshold: the output is the raw logit).

    FMPadPix (zero insertion) -> SWG_u (2x2, stride 1) -> MVAU_f (4*Cin -> Cout) -> Bias (ChannelwiseOp add)

Input H x W, output 2H x 2W. Balance is on FRAME cycles: pixels_node * cyc_per_pixel <= F; reported cyc_px is per OUTPUT pixel.
* The MVAU runs at OUTPUT resolution with 4*Cin*Cout MACs per output pixel, 3 of 4 window elements are inserted zeros (same lowering as the
  upsampling block's transposed conv).
* FMPadding_Pixel emits the (2H+1) x (2W+1) zero-inserted image: (2H+1)(2W+1) * ceil(Cin/SIMD) cycles. With SIMD = Cin that is 66,049 cycles for H = 128:
  together with the one-output-pixel-per-cycle MVAU floor (65,536) this is a floor on the whole network's frame budget, like the maxpool of the
  initial block.
* No thresholds, no residual, no join FIFO: only ordinary FIFOs.

Run: python3 fnl_block.py --cin 4 --cout 5 --bits 4 --height 128 --width 128 --F 73728 [--verify] [--rates] [--onnx PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bottleneck as reg  # noqa: E402
from bottleneck import (  # noqa: E402
    BottleneckResult, NodeResult, _SIZING_SCHEDULE, _dwc, _fifo_attrs, _geom, _min_pe, _search_mvau, finalize_fifo_costs,
)

fcm = reg.fcm


def model_fnl_block(
    cin: int, cout: int, bits: int, height: int, width: int, F: int | None = None, T_out: float | None = None, bias: bool = True,
) -> BottleneckResult:
    """cin -> cout final transposed conv (K = S = 2), INPUT map height x width, output 2*height x 2*width, uniform INT `bits`.
    Budget: F frame cycles, or T_out cycles per OUTPUT pixel (F = T_out * 4*height*width).
    bias=True (LayerQuantEnetFINN default, Int32Bias) adds the integer bias add after the MVAU; bias=False is the final layer of the production
    export (finn_enet_prod_export.py), which has none."""
    if (F is None) == (T_out is None):
        raise ValueError("give exactly one of F (frame cycles) or T_out (cycles per output pixel)")
    H, W, Ho, Wo = height, width, 2 * height, 2 * width
    px_in, px_out = H * W, Ho * Wo
    F = int(F if F is not None else math.floor(T_out * px_out))
    T, T_in, A = F / px_out, F / px_in, bits
    res = BottleneckResult(params=dict(
        cin=cin, cmid=cin, cout=cout, v=1, z=1, T=T, F=F, T_in=T_in, bits=bits, k=2, dilation=1, stride=2, height=H, width=W, hout=Ho, wout=Wo, pad=0,
        block="final", skip_pad="n/a", skip_order="n/a", pad_group=None, bias=bias,
    ))
    if px_out > F:
        raise ValueError(f"F={F} is below the final layer's floor of {px_out} cycles/frame (one output pixel per cycle)")

    g_f = _geom("final", cin, cout, Ho + 1, Wo + 1, Ho, Wo, k=2, s=1, d=1, p=0)       # lowered ConvTranspose (K=S=2) on the (2H+1)^2 image
    g_b = _geom("bias", cout, cout, Ho, Wo, Ho, Wo, op="Thresholding")
    pe_f, simd_f, c_f = _search_mvau(g_f, bits, F, extra_ok=lambda c: (Ho + 1) * (Wo + 1) * math.ceil(cin / c["simd_swu"]) <= F)
    # conv_cost_pe_simd prices a following standalone Thresholding (noActivation); the final layer has none
    c_f = dict(c_f, total_lut=c_f["total_lut"] - c_f["thr_lut"], thr_lut=0.0, thr_bram18=0.0, thr_uram18=0.0)
    pe_b = _min_pe(cout, px_out, F, "Bias") if bias else None
    c_b = fcm.stream_node_cost("add", g_b, pe_b) if bias else None   # ChannelwiseOp add: priced like AddStreams (provisional)
    swu = c_f["simd_swu"]
    cf = math.ceil(cin / swu)
    acc = c_f["acc_bits"]

    def row(name, op, pe, simd, frame_cycles, lut=0.0, bram=0.0, uram=0.0, dsp=0.0, in_w=0, out_w=0):
        return NodeResult(name=name, op=op, pe=pe, simd=simd, cyc_px=frame_cycles / px_out, frame_cycles=int(frame_cycles),
                          lut=lut, bram18=bram, uram=uram, dsp=dsp, in_width_bits=in_w, out_width_bits=out_w)

    nd = res.nodes
    nd.append(row("FMPadPix", "FMPadding_Pixel_hls", 0, swu, (Ho + 1) * (Wo + 1) * cf, in_w=swu * A, out_w=swu * A))
    nd.append(row("SWG_u", "ConvolutionInputGenerator_rtl 2x2", 0, swu, c_f["swu_cycles"], c_f["swu_lut"], c_f["swu_bram18"], c_f["swu_uram18"],
                  in_w=swu * A, out_w=swu * A))
    nd.append(row("MVAU_f", "MVAU rtl 2x2 (lowered ConvTranspose)", pe_f, simd_f, c_f["mvu_cycles"], c_f["mvu_lut"], c_f["wm_bram18"], c_f["wm_uram18"],
                  c_f["mvu_dsp"], in_w=simd_f * A, out_w=pe_f * acc))
    if bias:
        nd.append(row("Bias", "ChannelwiseOp_hls (bias add)", pe_b, 0, c_b["cycles"], c_b["total_lut"], in_w=pe_b * acc, out_w=pe_b * acc))
        _dwc("MVAU_f->Bias", pe_f * acc, pe_b * acc, cout, pe_f, pe_b, px_out, px_out, res.dwcs)
    for d in res.dwcs:
        if d.cyc_px > T + 1e-9:
            res.warnings.append(f"DWC {d.edge} needs {d.cyc_px:.2f} cyc/px (output pixels) > T_out={T:.2f}")
    res.skip_fifo = None
    res.latency_first_out_cycles = int(math.ceil(2 * (Wo + 1) * cf + nd[2].cyc_px + (cout / pe_b if bias else 0) + 8))
    res.frame_cycles = int(res.latency_first_out_cycles + (px_out - 1) * T)
    dwc_lut = sum(d.lut for d in res.dwcs)
    res.totals = dict(lut=sum(x.lut for x in nd) + dwc_lut, bram18=sum(x.bram18 for x in nd), uram=sum(x.uram for x in nd),
                      dsp=sum(x.dsp for x in nd), dwc_lut=dwc_lut)
    res.balance = dict(max_cyc_px=nd[2].cyc_px, min_cyc_px=nd[2].cyc_px, min_over_max=1.0, mean_util=nd[2].cyc_px / T, scope="MVAU_f")
    if max(x.frame_cycles for x in nd) > F:
        res.warnings.append(f"slowest node {max(x.frame_cycles for x in nd)} cycles/frame exceeds F={F}")
    res.warnings.append("FMPadding_Pixel and ChannelwiseOp LUTs are not calibrated in finn_cost_model (priced 0 / like AddStreams); the MVAU counts the MACs "
                        "on inserted zeros")
    return res


def verify_with_sim(r: BottleneckResult, fifo_depth: int = 2, tol: float = 0.02, max_tries: int = 6, shrink: bool = True,
                    tol_soft: float = 0.03, fifo_mem: str = "auto") -> dict:
    """Saturated 3-frame run; escalates the prefetch FIFO in front of FMPadPix and a uniform depth until the last frame's period per OUTPUT pixel is
    within tol of T_out, then shrinks every FIFO to its observed occupancy."""
    from fnl_block_sim import simulate_fnl

    p = r.params
    T = p["T"]
    cf = p["cin"] // next(x for x in r.nodes if x.name == "SWG_u").simd
    emap0 = {"FMPadPix": 2 * cf + 2}
    sat, tries, emap = None, 0, dict(emap0)
    for tries, (es, _, um) in enumerate(_SIZING_SCHEDULE[:max_tries], 1):
        depth = fifo_depth * um
        emap = {k: max(v * es, depth) for k, v in emap0.items()}
        sat = simulate_fnl(r, inject_interval=0, fifo_depth=depth, elastic_map=emap, frames=3)
        if not sat.deadlock and sat.steady_cyc_px <= T * (1 + tol):
            break
    ok = (not sat.deadlock) and sat.steady_cyc_px <= T * (1 + tol)
    if ok and shrink:
        small = simulate_fnl(r, inject_interval=0, fifo_depth=depth, elastic_map=emap,
                             fifo_depths={k: max(fifo_depth, occ) for k, occ in sat.fifo_max.items()}, frames=3)
        if not small.deadlock and small.steady_cyc_px <= T * (1 + tol):
            sat = small
    if not ok and not sat.deadlock and sat.steady_cyc_px <= T * (1 + tol_soft):
        r.warnings.append(f"steady {sat.steady_cyc_px:.2f} cyc/px is {sat.steady_cyc_px / T - 1:.1%} above T: residual per-frame window-fill gap")
        ok = True
    paced = simulate_fnl(r, inject_interval=p["T_in"], fifo_depth=depth, elastic_map=emap, fifo_depths=None, frames=2)
    graph = sat.graph
    for name, occ in sat.fifo_max.items():
        graph["fifos"][name]["max_occ"] = occ
    r.fifo_graph = graph
    r.verification = dict(
        ok=ok, steady_cyc_px=sat.steady_cyc_px, latency_first_out=paced.latency_first_out, frame_cycles=sat.cycles, uniform_depth=depth,
        tries=tries, deadlock=sat.deadlock, elastic_depth=emap.get("FMPadPix"), skip_needed_words=0, frame_periods=sat.frame_periods,
    )
    finalize_fifo_costs(r, fifo_mem)
    if not ok:
        raise RuntimeError(f"simulation does not reach T_out={T:.2f}: steady {sat.steady_cyc_px:.2f} cyc/px, deadlock={sat.deadlock}")
    return r.verification


def rate_report(r: BottleneckResult, with_sim: bool = True) -> str:
    from fnl_block_sim import simulate_fnl

    T = r.params["T"]
    n = {x.name: x for x in r.nodes}
    fr = {}
    if with_sim:
        if not r.verification:
            verify_with_sim(r)
        sim = simulate_fnl(r, inject_interval=0, fifo_depth=2, fifo_depths={k: f["depth"] for k, f in r.fifo_graph["fifos"].items()})
        fr = {name: sim.fractions(name, sim.steady_window) for name in sim.node_names if name in n}
    lines = [f"rate report, budget T_out = {T:.2f} cyc/output pixel (F = {r.params['F']} cycles/frame)", "",
             f"{'node':9s} {'cyc/px':>7s} {'util':>5s}" + ("   busy starv block" if fr else "")]
    for name in [x.name for x in r.nodes]:
        x = n[name]
        row = f"{name:9s} {x.cyc_px:7.2f} {x.cyc_px / T:5.2f}"
        if fr:
            f = fr[name]
            row += f"   {f['busy']:4.2f}  {f['starved']:4.2f}  {f['blocked']:4.2f}"
        lines.append(row)
    return "\n".join(lines)


_ROLE_OF = {"FMPadPix": "fmpadpix", "SWG_u": "swg_u", "MVAU_f": "mvau_f", "Bias": "bias"}


def to_folding_config(r: BottleneckResult) -> dict:
    """Role-keyed FINN nodeattrs + verified FIFO depths per edge + prediction (schema of bottleneck.to_folding_config). Needs verify_with_sim."""
    if not r.fifo_graph:
        raise RuntimeError("run verify_with_sim(result) before to_folding_config")
    p = r.params
    n = {x.name: x for x in r.nodes}
    fold = {"fmpadpix": {"SIMD": n["SWG_u"].simd}, "swg_u": {"SIMD": n["SWG_u"].simd, "parallel_window": int(n["MVAU_f"].simd > p["cin"])},
            "mvau_f": {"PE": n["MVAU_f"].pe, "SIMD": n["MVAU_f"].simd}}
    if "Bias" in n:
        fold["bias"] = {"PE": n["Bias"].pe}
    role = lambda name: "dwc" if name.startswith("DWC(") else _ROLE_OF.get(name, name.lower())
    fifos = []
    for name, f in r.fifo_graph["fifos"].items():
        if f["producer"] == "Source" or f["consumer"] == "Sink":
            continue
        fifos.append(dict(name=name, producer=role(f["producer"]), consumer=role(f["consumer"]), producer_node=f["producer"],
                          consumer_node=f["consumer"], depth=int(f["depth"]), width_bits=int(f["bits"]), max_occupancy=int(f.get("max_occ", 0)),
                          is_skip=False, is_join_fifo=False, **_fifo_attrs(r, name)))
    v = r.verification
    return dict(
        params=dict(p), folding=fold, fifos=fifos,
        predicted=dict(
            T=p["T"], F=p["F"], steady_cyc_px=v.get("steady_cyc_px"), latency_first_out=v.get("latency_first_out"), frame_cycles=v.get("frame_cycles"),
            skip_fifo_words=0, skip_fifo_width_bits=0, totals=dict(r.totals),
            nodes={_ROLE_OF[x.name]: dict(op=x.op, pe=x.pe, simd=x.simd, cyc_px=x.cyc_px, frame_cycles=x.frame_cycles, lut=x.lut, bram18=x.bram18,
                                          uram=x.uram, dsp=x.dsp, in_width_bits=x.in_width_bits, out_width_bits=x.out_width_bits) for x in r.nodes},
            dwcs=[dict(edge=d.edge, in_width=d.in_width, out_width=d.out_width, lut=d.lut) for d in r.dwcs],
        ),
    )


_OP_LABEL = {"FMPadPix": "FMPadding_Pixel_hls", "SWG_u": "ConvolutionInputGenerator_rtl", "MVAU_f": "MVAU_rtl", "Bias": "ChannelwiseOp_hls"}


def export_onnx(r: BottleneckResult, path: str) -> None:
    """Verified final-deconv dataflow graph for Netron (FINN op names, PE/SIMD/cycles/resources as attributes, DWCs, FIFOs)."""
    import onnx
    from onnx import TensorProto, helper

    if not r.fifo_graph:
        raise RuntimeError("run verify_with_sim(result) before export_onnx")
    p, g = r.params, r.fifo_graph
    H, W, Ho, Wo, cin, cout = p["height"], p["width"], p["hout"], p["wout"], p["cin"], p["cout"]
    by_name = {x.name: x for x in r.nodes}
    slowest = max(x.frame_cycles for x in r.nodes)
    node_shape = {"FMPadPix": (cin, Ho + 1, Wo + 1), "SWG_u": (4 * cin, Ho, Wo), "MVAU_f": (cout, Ho, Wo), "Bias": (cout, Ho, Wo)}
    last = r.nodes[-1].name
    ends = {"Source", "Sink"}
    fifos = {k: f for k, f in g["fifos"].items() if f["producer"] not in ends and f["consumer"] not in ends}
    shape = {"global_in": (cin, H, W)}
    onnx_nodes = []

    def out_tensor(node, fifo):
        outs = g["io"][node][1]
        return f"{node}:out" if len(outs) == 1 else f"{node}:out{outs.index(fifo)}"

    for name in g["nodes"]:
        if name in ends:
            continue
        ins, outs = g["io"][name]
        in_t = ["global_in" if g["fifos"][f]["producer"] == "Source" else out_tensor(g["fifos"][f]["producer"], f) for f in ins]
        if name in by_name:
            x = by_name[name]
            shp = node_shape[name]
            attrs = dict(stage="final_deconv", shape_CHW="x".join(map(str, shp)), pe=x.pe, simd=x.simd, cycles=int(x.frame_cycles),
                         ii_cycles_per_output_pixel=float(x.cyc_px), pct_of_budget=100 * x.frame_cycles / p["F"],
                         is_slowest_node=int(x.frame_cycles == slowest), lut=float(x.lut), bram18k=float(x.bram18), uram18=float(x.uram), dsp=int(x.dsp),
                         in_width_bits=x.in_width_bits, out_width_bits=x.out_width_bits, act_bits=p["bits"])
            op = _OP_LABEL[name]
        else:
            in_bits, out_bits = g["fifos"][ins[0]]["bits"], g["fifos"][outs[0]]["bits"]
            shp = shape[in_t[0]]
            attrs = dict(stage="final_deconv", shape_CHW="x".join(map(str, shp)), inWidth=in_bits, outWidth=out_bits,
                         lut=float(fcm.dwc_cost(in_bits, out_bits)["total_lut"]), bram18k=0.0, uram18=0.0, dsp=0)
            op = "StreamingDataWidthConverter_rtl"
        wired = []
        for fname, tin in zip(ins, in_t):
            if fname not in fifos:
                wired.append(tin)
                continue
            f = fifos[fname]
            depth, bits = f["depth"], f["bits"]
            onnx_nodes.append(helper.make_node(
                "StreamingFIFO_rtl", [tin], [f"{fname}:out"], name=fname, domain="finn_milp", stage="final_deconv", shape_CHW="x".join(map(str, shape[tin])),
                depth=int(depth), width_bits=int(bits), bits=int(depth * bits), max_occupancy=int(f.get("max_occ", 0)), producer=f["producer"],
                consumer=f["consumer"], **_fifo_attrs(r, fname)))
            shape[f"{fname}:out"] = shape[tin]
            wired.append(f"{fname}:out")
        out_names = [out_tensor(name, o) for o in outs]
        onnx_nodes.append(helper.make_node(op, wired, out_names, name=name, domain="finn_milp", **attrs))
        for t in out_names:
            shape[t] = shp
    final_out = out_tensor(last, g["io"][last][1][0])
    info = lambda t: helper.make_tensor_value_info(t, TensorProto.FLOAT, [1, *shape[t]])
    value_infos = [info(t) for t in shape if t not in ("global_in", final_out)]
    v = r.verification
    summary = (f"final deconv Cin={cin} Cout={cout} {H}x{W}->{Ho}x{Wo} INT{p['bits']} T_out={p['T']:.2f} F={p['F']} | LUT {r.totals['lut']:.0f} "
               f"BRAM18 {r.totals['bram18']:.1f} URAM {r.totals['uram']:.0f} DSP {r.totals['dsp']:.0f} | sim steady {v.get('steady_cyc_px', float('nan')):.2f} "
               f"cyc/px, latency {v.get('latency_first_out')} cycles")
    graph = helper.make_graph(onnx_nodes, "final_deconv_dataflow", [info("global_in")], [info(final_out)], value_info=value_infos, doc_string=summary)
    model = helper.make_model(graph, producer_name="fnl_block_analytical",
                              opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("finn_milp", 1)])
    onnx.save(model, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("cin", "cout", "bits", "height", "width"):
        ap.add_argument(f"--{a}", type=int, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--F", type=int, help="frame cycles budget")
    g.add_argument("--T-out", type=float, help="cycles per OUTPUT pixel")
    ap.add_argument("--no-bias", action="store_true", help="final layer without bias (production export)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--rates", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--onnx", metavar="PATH")
    ap.add_argument("--folding-json", metavar="PATH")
    a = ap.parse_args()
    r = model_fnl_block(a.cin, a.cout, a.bits, a.height, a.width, a.F, a.T_out, bias=not a.no_bias)
    if a.verify or a.rates or a.onnx or a.folding_json:
        verify_with_sim(r)
    if a.onnx:
        export_onnx(r, a.onnx)
    if a.folding_json:
        with open(a.folding_json, "w") as fh:
            json.dump(to_folding_config(r), fh, indent=2)
    print(json.dumps(r.to_dict(), indent=2) if a.json else r.report())
    if a.rates:
        print()
        print(rate_report(r))
    if a.onnx:
        print(f"wrote {a.onnx}")


if __name__ == "__main__":
    main()
