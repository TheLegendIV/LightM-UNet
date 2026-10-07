"""Check whether the folding config that build_partition_folding_config *computed* for
partitions 5/6 (hawq_folding_config_partition{5,6}.json) actually matches the nodeattrs that
landed in the real built graph (partition_{5,6}_prefifo_autosize.onnx). A mismatch here means
FINN's own folding-apply step (or a later transform) silently changed PE/SIMD/parallel_window
after the MILP's folding config was applied -- which would change that node's real cycles/II,
invalidating the MILP's FIFO depth calc for every edge touching it (the MILP's depths assume
specific per-node cycle counts derived from the SAME pe/simd it put in the folding config).
"""
import json
import sys

import onnx

ATTR_NAMES = ("PE", "SIMD", "parallel_window", "depth_trigger_bram")


def node_attrs(node):
    out = {}
    for a in node.attribute:
        if a.name in ATTR_NAMES:
            if a.type == onnx.AttributeProto.INT:
                out[a.name] = a.i
            elif a.type == onnx.AttributeProto.STRING:
                out[a.name] = a.s.decode()
    return out


def check(part_idx, folding_path, onnx_path):
    with open(folding_path) as f:
        folding = json.load(f)
    folding.pop("Defaults", None)

    model = onnx.load(onnx_path)
    by_name = {n.name: n for n in model.graph.node}

    print(f"=== partition {part_idx}: {len(folding)} folding entries ===")
    n_ok = n_missing = n_mismatch = 0
    for node_name, wanted in folding.items():
        node = by_name.get(node_name)
        if node is None:
            print(f"  MISSING NODE: {node_name} wanted={wanted}")
            n_missing += 1
            continue
        actual = node_attrs(node)
        diffs = {}
        for k, v in wanted.items():
            av = actual.get(k)
            if av is None:
                continue  # attr not applicable/present on this op_type, skip
            if av != v:
                diffs[k] = (v, av)
        if diffs:
            print(f"  MISMATCH {node_name} ({node.op_type}): wanted={wanted} actual={actual} diffs={diffs}")
            n_mismatch += 1
        else:
            n_ok += 1
    print(f"  -> ok={n_ok} missing={n_missing} mismatch={n_mismatch}")
    return n_missing, n_mismatch


if __name__ == "__main__":
    base = r"c:\DEV\repos\LightM-UNet\hardware\temp\p5_p6_diag"
    total_missing = total_mismatch = 0
    for idx in (5, 6):
        m, mm = check(
            idx,
            f"{base}\\hawq_folding_config_partition{idx}.json",
            f"{base}\\partition_{idx}_prefifo_autosize.onnx",
        )
        total_missing += m
        total_mismatch += mm
    print(f"\nTOTAL missing={total_missing} mismatch={total_mismatch}")
    sys.exit(1 if (total_missing or total_mismatch) else 0)
