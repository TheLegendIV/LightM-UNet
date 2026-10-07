"""For each DuplicateStreams/AddStreams fork-join pair in a partition, walk both
branches node-by-node and print get_exp_cycles() (FINN's own per-node expected
cycle-count estimate from folding config) to find which node is the throughput
outlier causing a branch imbalance.

Run INSIDE the FINN container (HOME=/tmp/home_dir):
    HOME=/tmp/home_dir python3 branch_cycle_compare.py <stitched.onnx>
"""
import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp


def build_maps(model):
    producer_of = {}
    consumers_of = {}
    for node in model.graph.node:
        for out in node.output:
            producer_of[out] = node
        for inp in node.input:
            consumers_of.setdefault(inp, []).append(node)
    return producer_of, consumers_of


def exp_cycles(node):
    try:
        inst = getCustomOp(node)
        return inst.get_exp_cycles()
    except Exception as e:
        return f"ERR:{e}"


def trace_branch(start_node, stop_op_type, consumers_of):
    """Walk forward along a single-consumer chain until hitting a node of
    stop_op_type (exclusive) or a fan-out/fan-in point."""
    path = [start_node]
    cur = start_node
    seen = {start_node.name}
    for _ in range(200):
        outs = []
        for out_t in cur.output:
            outs.extend(consumers_of.get(out_t, []))
        if len(outs) != 1:
            break
        nxt = outs[0]
        if nxt.op_type == stop_op_type:
            break
        if nxt.name in seen:
            break
        seen.add(nxt.name)
        path.append(nxt)
        cur = nxt
    return path


def main():
    model = ModelWrapper(sys.argv[1])
    producer_of, consumers_of = build_maps(model)

    forks = [n for n in model.graph.node if n.op_type == "DuplicateStreams_hls"]
    for fork in forks:
        print(f"\n=== Fork: {fork.name} ===")
        out_tensors = list(fork.output)
        for i, out_t in enumerate(out_tensors):
            branch_consumers = consumers_of.get(out_t, [])
            if not branch_consumers:
                continue
            start = branch_consumers[0]
            path = trace_branch(start, "AddStreams_hls", consumers_of)
            total = 0
            print(f"  -- branch {i} --")
            for node in path:
                c = exp_cycles(node)
                try:
                    total += int(c)
                except (TypeError, ValueError):
                    pass
                print(f"     {node.op_type:32s} {node.name:32s} exp_cycles={c}")
            print(f"     TOTAL exp_cycles (sum over branch) = {total}")


if __name__ == "__main__":
    main()
