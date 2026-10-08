"""Print differences between two MILP layer_bits_folding_final.json files (FIFO depth rows + per_layer folds)."""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
print("keys:", sorted(a), "|", sorted(b) if sorted(b) != sorted(a) else "same")
for sec in ("per_layer", "extra_nodes"):
    x, y = a.get(sec, {}), b.get(sec, {})
    diff = [k for k in sorted(set(x) | set(y)) if x.get(k) != y.get(k)]
    print(f"{sec}: {len(diff)} differing of {len(set(x) | set(y))}", diff[:10])
for sec in ("inter_block_fifos", "intra_block_fifos"):
    x, y = a.get(sec), b.get(sec)
    if isinstance(x, dict):
        diff = [k for k in sorted(set(x) | set(y or {})) if x.get(k) != (y or {}).get(k)]
        print(f"{sec}: {len(diff)} differing")
        for k in diff[:20]:
            xv, yv = x.get(k), (y or {}).get(k)
            print("  ", k, (xv.get("depth") if isinstance(xv, dict) else xv), "->", (yv.get("depth") if isinstance(yv, dict) else yv))
    elif isinstance(x, list):
        key = lambda r: r.get("name")
        xm, ym = {key(r): r for r in x}, {key(r): r for r in (y or [])}
        diff = [k for k in sorted(set(xm) | set(ym)) if (xm.get(k) or {}).get("depth") != (ym.get(k) or {}).get("depth")]
        print(f"{sec}: {len(diff)} differing")
        for k in diff[:20]:
            print("  ", k, (xm.get(k) or {}).get("depth"), "->", (ym.get(k) or {}).get("depth"))
