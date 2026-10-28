"""Whole-network dataflow graph as ONNX, for a visual check in Netron (companion of net_fold.py / net_fifo.py).

One node per hardware node of every analytical block (FINN op names; PE / SIMD / cycles / LUT / BRAM / DSP as attributes), the DWCs the block models
insert, every intra-block StreamingFIFO the block verification kept (depth > 2: FINN's RemoveShallowFIFOs deletes the others), and one
StreamingFIFO_rtl at the output of each block (the inter-block FIFOs sized by net_fifo; drawn even at depth 2 to mark the block boundary, flagged). Node and tensor names are prefixed with the block name.
It is a picture of the design, not an executable model: the domain is `finn_milp`, tensors have no shapes.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # MILP/ (node_names)
import finn_cost_model as fcm  # noqa: E402
from bottleneck import fifo_memory  # noqa: E402
from node_names import block_output_name, milp_role, table_kind  # noqa: E402

DOMAIN = "finn_milp"


def _op_type(op: str) -> str:
    first = op.split(" ")[0]
    return "MVAU_rtl" if first == "MVAU" else first


def _fifo_node(helper, name, src, dst, f, extra=None, stage=""):
    depth, bits = int(f["depth"]), int(f["bits"])
    mem = fifo_memory(bits, depth)
    return helper.make_node(
        "StreamingFIFO_rtl", [src], [dst], name=name, domain=DOMAIN, stage=stage, depth=depth, width_bits=bits, bits=depth * bits,
        max_occupancy=int(f.get("max_occ", 0)), mem=mem["mem"], depth_alloc=int(mem["depth_alloc"]), mem_lut=int(mem["lut"]),
        mem_bram18=int(mem["bram18"]), mem_uram=int(mem["uram"]), producer=str(f.get("producer", "")), consumer=str(f.get("consumer", "")),
        **(extra or {}),
    )


def export_net_onnx(blocks: list, interfaces: list, path: str, summary: str = "") -> dict:
    """blocks = [(stage, kind, verified result)] in dataflow order; interfaces = [dict(depth, width_bits, max_occ, ...)] between consecutive blocks
    (len(blocks) - 1; the last block's output is the network output). Returns node / FIFO counts."""
    import onnx
    from onnx import TensorProto, helper

    nodes, n_fifo, n_dwc = [], 0, 0
    block_in = "global_in"
    out_tensor = None
    prev_output = None
    for i, (stage, kind, r) in enumerate(blocks):
        role = lambda node: milp_role(stage, table_kind(kind, r.params), node, prev_output)
        g = r.fifo_graph
        by_name = {n.name: n for n in r.nodes}
        act_bits = r.params.get("bits")
        src_fifo = next(n for n, f in g["fifos"].items() if f["producer"] == "Source")
        sink_fifo = next(n for n, f in g["fifos"].items() if f["consumer"] == "Sink")

        def out_t(node, fifo):
            outs = g["io"][node][1]
            return f"{stage}/{node}:out" if len(outs) == 1 else f"{stage}/{node}:out{outs.index(fifo)}"

        for name in g["nodes"]:
            if name in ("Source", "Sink"):
                continue
            ins, outs = g["io"][name]
            wired = []
            for fn in ins:
                f = g["fifos"][fn]
                if f["producer"] == "Source":
                    wired.append(block_in)
                    continue
                t = out_t(f["producer"], fn)
                if f["depth"] > 2:
                    helper_t = f"{stage}/{fn}:fifo"
                    nodes.append(_fifo_node(helper, f"{role(f['producer'])}=>{role(f['consumer'])}", t, helper_t, f, stage=stage,
                                            extra=dict(producer_milp=role(f["producer"]), consumer_milp=role(f["consumer"]), analytical_name=fn)))
                    n_fifo += 1
                    t = helper_t
                wired.append(t)
            out_names = [out_t(name, o) for o in outs]
            if name in by_name:
                n = by_name[name]
                attrs = dict(
                    stage=stage, block_kind=kind, description=n.op, pe=int(n.pe), simd=int(n.simd), cycles=int(n.frame_cycles), ii_cycles_per_pixel=float(n.cyc_px),
                    lut=float(n.lut), bram18k=float(n.bram18), uram18=float(n.uram), dsp=int(n.dsp), in_width_bits=int(n.in_width_bits),
                    out_width_bits=int(n.out_width_bits), act_bits=int(act_bits or 0),
                )
                op = _op_type(n.op)
            else:                                   # DWC(prev->next): widths are those of the FIFOs on either side
                ib, ob = g["fifos"][ins[0]]["bits"], g["fifos"][outs[0]]["bits"]
                attrs = dict(stage=stage, block_kind=kind, inWidth=int(ib), outWidth=int(ob), lut=float(fcm.dwc_cost(ib, ob)["total_lut"]),
                             bram18k=0.0, uram18=0.0, dsp=0)
                op = "StreamingDataWidthConverter_rtl"
                n_dwc += 1
            nodes.append(helper.make_node(op, wired, out_names, name=role(name), domain=DOMAIN, analytical_name=name, **attrs))
        # block output -> inter-block FIFO -> next block
        f_sink = g["fifos"][sink_fifo]
        out_tensor = out_t(f_sink["producer"], sink_fifo)
        prev_output = block_output_name(stage, kind)           # the next block's Dup is named after this one
        if i < len(blocks) - 1:
            itf = interfaces[i]
            nxt = blocks[i + 1]
            f_next = nxt[2].fifo_graph["fifos"][next(n for n, f in nxt[2].fifo_graph["fifos"].items() if f["producer"] == "Source")]
            f_if = dict(depth=itf["depth"], bits=itf["width_bits"], max_occ=itf.get("max_occ", 0), producer=f"{stage}", consumer=nxt[0])
            t = f"iface{i}:{stage}->{nxt[0]}"
            nodes.append(_fifo_node(helper, f"{stage}->{nxt[0]}", out_tensor, t, f_if,       # always drawn: it marks the block boundary
                                    dict(inter_block=1, producer_milp=block_output_name(stage, kind), consumer_milp=nxt[0], period_cycles=int(itf.get("period") or 0), removed_by_finn_if_depth_le_2=int(itf["depth"] <= 2)), stage=stage))
            n_fifo += 1
            if int(f_next["bits"]) != int(f_sink["bits"]):
                t2 = f"iface{i}:dwc"
                nodes.append(helper.make_node(
                    "StreamingDataWidthConverter_rtl", [t], [t2], name=f"{stage}->{nxt[0]}/DWC", domain=DOMAIN, stage=stage, block_kind="interface",
                    inWidth=int(f_sink["bits"]), outWidth=int(f_next["bits"]), lut=float(fcm.dwc_cost(f_sink["bits"], f_next["bits"])["total_lut"]),
                    bram18k=0.0, uram18=0.0, dsp=0))
                n_dwc += 1
                t = t2
            block_in = t
    info = lambda t: helper.make_tensor_value_info(t, TensorProto.FLOAT, ["stream"])
    graph = helper.make_graph(nodes, "enet_dataflow", [info("global_in")], [info(out_tensor)], doc_string=summary)
    model = helper.make_model(graph, producer_name="net_fold", opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(DOMAIN, 1)])
    onnx.save(model, path)
    return dict(nodes=len(nodes), fifos=n_fifo, dwcs=n_dwc)
