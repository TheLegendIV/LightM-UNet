"""Rank StreamingFIFO_rtl nodes across all 8 v2 partitions by depth and by
total bit-storage (depth * dataType_bits * folded_shape[-1] parallelism),
to find where FIFO resource cost is concentrated.
Usage: python analyze_fifo_depths.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
BITS_RE = re.compile(r"(\d+)$")


def dtype_bits(dtype: str) -> int:
    m = BITS_RE.search(dtype)
    return int(m.group(1)) if m else 0


rows = []
for i in range(8):
    path = HERE / f"partition_{i}_fifos.json"
    for entry in json.loads(path.read_text()):
        attrs = entry["attrs"]
        depth = attrs.get("depth", 0)
        dtype = attrs.get("dataType", "")
        bits = dtype_bits(dtype)
        folded_shape = attrs.get("folded_shape", [])
        parallelism = folded_shape[-1] if folded_shape else 1
        width_bits = bits * parallelism
        rows.append({
            "partition": i,
            "node_name": entry["node_name"],
            "depth": depth,
            "impl_style": attrs.get("impl_style"),
            "ram_style": attrs.get("ram_style"),
            "dataType": dtype,
            "folded_shape": folded_shape,
            "width_bits": width_bits,
            "total_bits": depth * width_bits,
        })

print(f"Total FIFO nodes: {len(rows)}")
print(f"Sum of all depths: {sum(r['depth'] for r in rows)}")
print(f"Sum of all total_bits: {sum(r['total_bits'] for r in rows):,}  ({sum(r['total_bits'] for r in rows)/8/1024:.1f} KB)")

by_partition = {}
for r in rows:
    p = by_partition.setdefault(r["partition"], {"count": 0, "depth_sum": 0, "bits_sum": 0})
    p["count"] += 1
    p["depth_sum"] += r["depth"]
    p["bits_sum"] += r["total_bits"]

print("\n=== Per-partition totals ===")
print(f"{'Part':<5}{'#FIFOs':<8}{'SumDepth':<10}{'SumBits':<14}{'SumKB':<10}")
for p in sorted(by_partition):
    d = by_partition[p]
    print(f"{p:<5}{d['count']:<8}{d['depth_sum']:<10}{d['bits_sum']:<14,}{d['bits_sum']/8/1024:<10.1f}")

print("\n=== Top 20 FIFOs by DEPTH ===")
for r in sorted(rows, key=lambda r: -r["depth"])[:20]:
    print(f"p{r['partition']} {r['node_name']:<28} depth={r['depth']:<6} dtype={r['dataType']:<8} "
          f"folded_shape={r['folded_shape']} impl={r['impl_style']}/{r['ram_style']}")

print("\n=== Top 20 FIFOs by TOTAL BITS (depth x width) ===")
for r in sorted(rows, key=lambda r: -r["total_bits"])[:20]:
    print(f"p{r['partition']} {r['node_name']:<28} depth={r['depth']:<6} width_bits={r['width_bits']:<6} "
          f"total_bits={r['total_bits']:<10,} ({r['total_bits']/8/1024:.2f} KB) dtype={r['dataType']:<8} impl={r['impl_style']}/{r['ram_style']}")

print("\n=== impl_style breakdown ===")
impl_counts = {}
for r in rows:
    key = (r["impl_style"], r["ram_style"])
    impl_counts[key] = impl_counts.get(key, 0) + 1
for key, count in sorted(impl_counts.items(), key=lambda kv: -kv[1]):
    print(f"{key}: {count}")
