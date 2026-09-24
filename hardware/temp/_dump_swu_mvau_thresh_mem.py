"""Standalone: dump SWU (ConvolutionInputGenerator_rtl) + MVAU/VVAU (weight
memory) + Thresholding BRAM18K/URAM estimates for the 8 finished
partition_N_fifo_sized.onnx files (512x512 case), using each node's own
bram_estimation()/uram_estimation() methods -- no formulas reimplemented here.
"""
import csv
import glob
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

COMPUTE_OP_TYPES = ["ConvolutionInputGenerator_rtl", "MVAU_rtl", "VVAU_rtl", "Thresholding_rtl"]

PARTITIONS_DIR = (
    "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
    "12_dense_relu_warmstart150ep_alpha025_fifo_depths_only_512x512_20260923_120832/"
    "intermediate_models/supported_op_partitions"
)

rows = []
paths = sorted(glob.glob(os.path.join(PARTITIONS_DIR, "partition_*_fifo_sized.onnx")))
print(f"Found {len(paths)} finished partition(s): {[os.path.basename(p) for p in paths]}")

for path in paths:
    partition = os.path.basename(path).split("_")[1]
    model = ModelWrapper(path)
    for node in model.graph.node:
        if node.op_type not in COMPUTE_OP_TYPES:
            continue
        inst = getCustomOp(node)
        bram18k = inst.bram_estimation()
        uram = inst.uram_estimation()
        lut = inst.lut_estimation()
        try:
            ram_style = inst.get_nodeattr("ram_style")
        except Exception:
            ram_style = ""
        try:
            mem_mode = inst.get_nodeattr("mem_mode")
        except Exception:
            mem_mode = ""
        rows.append({
            "partition": partition, "node_name": node.name, "op_type": node.op_type,
            "mem_mode": mem_mode, "ram_style": ram_style,
            "bram18k": bram18k, "uram18k": uram, "lut": lut,
        })

out_csv = "/home/thelegendiv/finn/notebooks/enet/swu_mvau_thresh_mem_512x512.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nWrote {len(rows)} entries to {out_csv}")

# Summary per partition x op_type
print(f"\n{'partition':<10}{'op_type':<32}{'count':>7}{'bram18k':>10}{'uram18k':>10}{'lut':>10}")
by_key = {}
for r in rows:
    key = (r["partition"], r["op_type"])
    by_key.setdefault(key, []).append(r)
for (p, op), rs in sorted(by_key.items()):
    print(f"{p:<10}{op:<32}{len(rs):>7}{sum(r['bram18k'] for r in rs):>10}"
          f"{sum(r['uram18k'] for r in rs):>10}{sum(r['lut'] for r in rs):>10}")

print(f"\n{'partition':<10}{'total_bram18k':>15}{'total_uram18k':>15}")
by_part = {}
for r in rows:
    by_part.setdefault(r["partition"], []).append(r)
grand_bram = grand_uram = 0
for p, rs in sorted(by_part.items()):
    tb = sum(r["bram18k"] for r in rs)
    tu = sum(r["uram18k"] for r in rs)
    grand_bram += tb
    grand_uram += tu
    print(f"{p:<10}{tb:>15}{tu:>15}")

print(f"\nGRAND TOTAL (SWU+MVAU/VVAU+Thresholding, all 8 partitions): "
      f"{grand_bram} BRAM18K, {grand_uram} URAM")
print("ZCU7EV budget: 984 BRAM18K, 128 URAM")

# breakdown by op type across all partitions
print("\nBreakdown by op_type (all partitions):")
by_op = {}
for r in rows:
    by_op.setdefault(r["op_type"], []).append(r)
for op, rs in sorted(by_op.items()):
    print(f"  {op:<32} count={len(rs):<5} bram18k={sum(r['bram18k'] for r in rs):<8} "
          f"uram18k={sum(r['uram18k'] for r in rs):<8} lut={sum(r['lut'] for r in rs)}")
