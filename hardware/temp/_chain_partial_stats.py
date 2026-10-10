import re, sys
import numpy as np
rows = []
pat = re.compile(r"case(\d+): agreement\(rtl vs torch\) ([\d.]+) fg-dice rtl ([\d.nan]+) torch ([\d.nan]+) classes rtl (\[.*?\]) torch (\[.*?\]) gt (\[.*?\])")
for l in open("/tmp/chain_mixed_all.log"):
    m = pat.search(l)
    if m:
        rows.append((int(m[1]), float(m[2]), float(m[3]), float(m[4]), eval(m[5]), eval(m[6]), eval(m[7])))
a = np.array([r[1] for r in rows]); dr = np.array([r[2] for r in rows]); dt = np.array([r[3] for r in rows])
print(f"n={len(rows)} agree mean {a.mean():.6f} min {a.min():.6f} | dice rtl {np.nanmean(dr):.4f} torch {np.nanmean(dt):.4f} drop {np.nanmean(dt)-np.nanmean(dr):.4f}")
print("agree quantiles 1/10/50/90:", np.percentile(a, [1, 10, 50, 90]).round(5))
big = [(r[0], round(r[3] - r[2], 3), r[4], r[5], r[6]) for r in rows if r[3] - r[2] > 0.05]
print(f"cases with dice drop > 0.05: {len(big)}")
# spurious class: rtl has pixels of a class where torch has none
spur = 0
for r in rows:
    if any(rc > 0 and tc == 0 for rc, tc in zip(r[4][1:], r[5][1:])):
        spur += 1
print("cases where RTL predicts a fg class that torch never predicts:", spur)
missing = sum(1 for r in rows if any(rc == 0 and tc > 0 for rc, tc in zip(r[4][1:], r[5][1:])))
print("cases where RTL misses a fg class that torch predicts:", missing)
for b in big[:12]:
    print(b)
# dice drop excluding spurious-class cases
ok = [r for r in rows if not any(rc > 0 and tc == 0 for rc, tc in zip(r[4][1:], r[5][1:])) and not any(rc == 0 and tc > 0 for rc, tc in zip(r[4][1:], r[5][1:]))]
d1 = np.array([r[3] - r[2] for r in ok])
print(f"cases without class appear/disappear: {len(ok)} mean dice drop {np.nanmean(d1):.4f}")
