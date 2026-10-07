"""Compare StreamingFIFO depths between the analytical (MILP-forced, 'refix1')
and FINN-autosize ('autosize1') partition 6 builds, and dump the local graph
neighborhood (producer/consumer op_types) around every node whose name
contains 'MVAU_rtl_8' to check for a fork/join (DuplicateStreams ... AddStreams)
structure around it.

Run INSIDE the FINN container (HOME=/tmp/home_dir):
    HOME=/tmp/home_dir python3 compare_fifo_depths_partition6.py \
        <analytical.onnx> <autosize.onnx> <out.json>
"""
import json
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp


def build_edge_maps(model):
    producer_of = {}  # tensor_name -> node
    consumers_of = {}  # tensor_name -> [node, ...]
    for node in model.graph.node:
        for out in node.output:
            producer_of[out] = node
        for inp in node.input:
            consumers_of.setdefault(inp, []).append(node)
    return producer_of, consumers_of


def fifo_rows(model):
    producer_of, consumers_of = build_edge_maps(model)
    rows = []
    for node in model.graph.node:
        if "StreamingFIFO" not in node.op_type:
            continue
        inst = getCustomOp(node)
        depth = None
        for key in ("depth", "impl_style"):
            try:
                if key == "depth":
                    depth = inst.get_nodeattr(key)
            except Exception:
                pass
        impl_style = None
        try:
            impl_style = inst.get_nodeattr("impl_style")
        except Exception:
            pass
        in_tensor = node.input[0]
        out_tensor = node.output[0]
        prod = producer_of.get(in_tensor)
        cons_list = consumers_of.get(out_tensor, [])
        rows.append({
            "fifo_name": node.name,
            "depth": depth,
            "impl_style": impl_style,
            "producer_name": prod.name if prod is not None else None,
            "producer_op": prod.op_type if prod is not None else None,
            "consumer_names": [c.name for c in cons_list],
            "consumer_ops": [c.op_type for c in cons_list],
        })
    return rows


def neighborhood(model, name_substr, radius=2):
    producer_of, consumers_of = build_edge_maps(model)
    name_to_node = {n.name: n for n in model.graph.node}
    hits = [n for n in model.graph.node if name_substr in n.name]
    out = []
    for h in hits:
        entry = {"node": h.name, "op_type": h.op_type, "upstream": [], "downstream": []}
        # walk upstream `radius` hops
        frontier = [h]
        seen = set()
        for _ in range(radius):
            new_frontier = []
            for node in frontier:
                for inp in node.input:
                    prod = producer_of.get(inp)
                    if prod is not None and prod.name not in seen:
                        seen.add(prod.name)
                        entry["upstream"].append({"name": prod.name, "op_type": prod.op_type})
                        new_frontier.append(prod)
            frontier = new_frontier
        frontier = [h]
        seen = set()
        for _ in range(radius):
            new_frontier = []
            for node in frontier:
                for out_t in node.output:
                    for cons in consumers_of.get(out_t, []):
                        if cons.name not in seen:
                            seen.add(cons.name)
                            entry["downstream"].append({"name": cons.name, "op_type": cons.op_type})
                            new_frontier.append(cons)
            frontier = new_frontier
        out.append(entry)
    return out


def main():
    analytical_path, autosize_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    analytical = ModelWrapper(analytical_path)
    autosize = ModelWrapper(autosize_path)

    analytical_fifos = {r["fifo_name"]: r for r in fifo_rows(analytical)}
    autosize_fifos = {r["fifo_name"]: r for r in fifo_rows(autosize)}

    all_names = sorted(set(analytical_fifos) | set(autosize_fifos))
    comparison = []
    for name in all_names:
        a = analytical_fifos.get(name)
        b = autosize_fifos.get(name)
        comparison.append({
            "fifo_name": name,
            "analytical_depth": a["depth"] if a else None,
            "autosize_depth": b["depth"] if b else None,
            "producer_op": (b or a)["producer_op"],
            "consumer_ops": (b or a)["consumer_ops"],
            "only_in": "both" if (a and b) else ("analytical" if a else "autosize"),
        })

    mvau8_analytical = neighborhood(analytical, "MVAU_rtl_8", radius=10)
    mvau8_autosize = neighborhood(autosize, "MVAU_rtl_8", radius=10)

    result = {
        "fifo_comparison": comparison,
        "mvau8_neighborhood_analytical": mvau8_analytical,
        "mvau8_neighborhood_autosize": mvau8_autosize,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote comparison for {len(comparison)} fifos to {out_path}")
    print(f"MVAU_rtl_8-matching nodes: analytical={len(mvau8_analytical)} autosize={len(mvau8_autosize)}")


if __name__ == "__main__":
    main()
