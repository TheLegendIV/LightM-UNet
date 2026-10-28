import json
import sys
from pathlib import Path

root = Path(r"C:\DEV\repos\LightM-UNet\ILP\artifacts\S12_512_bilin_int8_maxfps")
files = [
    root / "maxfps" / "layer_bits_folding_maxfps.json",
    root / "pinned_minres" / "layer_bits_folding_pinned_minres.json",
    root / "simfifo_pairs" / "layer_bits_folding_S12_512_bilin_int8_maxfps_pairs.json",
]
for f in files:
    if not f.exists():
        print("missing", f)
        continue
    j = json.load(open(f))
    rows = []
    for n, e in j["per_layer"].items():
        c = e.get("cycles")
        if c is not None:
            rows.append((c, n, e.get("mvu_cycles"), e.get("swu_cycles"), e.get("impl_style")))
    for n, e in j.get("extra_nodes", {}).items():
        c = e.get("cycles")
        if c is not None and e.get("kind") not in ("dup",):
            rows.append((c, n, None, None, e.get("kind")))
    rows.sort(key=lambda r: r[0], reverse=True)
    print(f"== {f.parent.name}/{f.name} ({len(rows)} nodes)  status={j.get('status')}")
    diag = j.get("_diagnostics", {})
    for k in ("frame_cycles", "target_fps", "max_cycles", "fps", "bottleneck_cycles", "critical_cycles"):
        if k in diag:
            print("   diag", k, diag[k])
    for r in rows[:10]:
        print(f"   {r[0]:>9}  {r[1]:34s} mvu={r[2]} swu={r[3]} {r[4]}")
    pools = [r for r in rows if r[1].endswith(".pool")]
    print("   pools:", [(r[1], r[0]) for r in pools])

