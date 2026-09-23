"""Standalone: dump FIFO depths/widths + BRAM18K estimates for whichever
partition_N_fifo_sized.onnx files already exist (doesn't need all 8 done).
Uses FINN's own StreamingFIFO bram_estimation()/uram_estimation()/lut_estimation()
formulas directly (see finn/src/finn/custom_op/fpgadataflow/streamingfifo.py)."""
import csv
import glob
import math
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

FIFO_OP_TYPES = ["StreamingFIFO_rtl", "StreamingFIFO_hls", "StreamingFIFO"]

PARTITIONS_DIR = (
    "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
    "12_dense_relu_warmstart150ep_alpha025_fifo_depths_only_512x512_20260923_120832/"
    "intermediate_models/supported_op_partitions"
)


def bram18k_estimate(depth, W, ram_style):
    """Exact port of StreamingFIFO.bram_estimation() (impl_style=vivado,
    ram_style=block/auto only -- rtl/distributed/ultra FIFOs use 0 BRAM18K)."""
    if ram_style not in ("block", "auto"):
        return 0
    if W == 1:
        return math.ceil(depth / 16384)
    elif W == 2:
        return math.ceil(depth / 8192)
    elif W <= 4:
        return math.ceil(depth / 4096) * math.ceil(W / 4)
    elif W <= 9:
        return math.ceil(depth / 2048) * math.ceil(W / 9)
    elif W <= 18 or depth > 512:
        return math.ceil(depth / 1024) * math.ceil(W / 18)
    else:
        return math.ceil(depth / 512) * math.ceil(W / 36)


def uram_estimate(depth, W, ram_style):
    if ram_style != "ultra":
        return 0
    return math.ceil(depth / 4096) * math.ceil(W / 72)


rows = []
paths = sorted(glob.glob(os.path.join(PARTITIONS_DIR, "partition_*_fifo_sized.onnx")))
print(f"Found {len(paths)} finished partition(s): {[os.path.basename(p) for p in paths]}")

for path in paths:
    partition = os.path.basename(path).split("_")[1]
    model = ModelWrapper(path)
    for node in model.graph.node:
        if node.op_type not in FIFO_OP_TYPES:
            continue
        inst = getCustomOp(node)
        depth = inst.get_nodeattr("depth")
        impl_style = inst.get_nodeattr("impl_style")
        ram_style = inst.get_nodeattr("ram_style")
        dtype_name = inst.get_nodeattr("dataType")
        dtype = DataType[dtype_name]
        bitwidth = dtype.bitwidth()
        folded_shape = inst.get_nodeattr("folded_shape")
        elems_per_beat = folded_shape[-1]
        W = elems_per_beat * bitwidth
        size_elems = elems_per_beat * depth
        size_bits = size_elems * bitwidth
        bram18k = bram18k_estimate(depth, W, ram_style) if impl_style == "vivado" else 0
        uram = uram_estimate(depth, W, ram_style) if impl_style == "vivado" else 0
        rows.append({
            "partition": partition, "node_name": node.name, "impl_style": impl_style,
            "ram_style": ram_style, "dataType": dtype_name, "bitwidth": bitwidth,
            "depth": depth, "elems_per_beat": elems_per_beat, "width_bits": W,
            "size_bits": size_bits, "size_bytes": size_bits / 8,
            "bram18k": bram18k, "uram18k": uram,
        })

out_csv = "/home/thelegendiv/finn/notebooks/enet/fifo_depths_partial_512x512.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nWrote {len(rows)} FIFO entries to {out_csv}")

# Summary per partition
print(f"\n{'partition':<10}{'n_fifos':>8}{'max_depth':>10}{'total_bram18k':>15}{'total_uram18k':>15}{'total_KB':>10}")
by_part = {}
for r in rows:
    by_part.setdefault(r["partition"], []).append(r)
for p, rs in sorted(by_part.items()):
    max_depth = max(r["depth"] for r in rs)
    total_bram = sum(r["bram18k"] for r in rs)
    total_uram = sum(r["uram18k"] for r in rs)
    total_kb = sum(r["size_bytes"] for r in rs) / 1024
    print(f"{p:<10}{len(rs):>8}{max_depth:>10}{total_bram:>15}{total_uram:>15}{total_kb:>10.1f}")

grand_bram = sum(r["bram18k"] for r in rows)
grand_uram = sum(r["uram18k"] for r in rows)
grand_kb = sum(r["size_bytes"] for r in rows) / 1024
print(f"\nGRAND TOTAL (these {len(by_part)} partitions only): {grand_bram} BRAM18K, {grand_uram} URAM, {grand_kb:.1f} KB")
print("ZCU7EV budget: 984 BRAM18K, 128 URAM -- for reference (this is a PARTIAL total, 4/8 partitions)")

# Top 10 deepest FIFOs across finished partitions
print("\nTop 10 deepest FIFOs so far:")
for r in sorted(rows, key=lambda r: -r["depth"])[:10]:
    print(f"  partition {r['partition']:<3} {r['node_name']:<30} depth={r['depth']:<8} "
          f"W={r['width_bits']:<5} impl={r['impl_style']:<8} ram={r['ram_style']:<12} bram18k={r['bram18k']}")
