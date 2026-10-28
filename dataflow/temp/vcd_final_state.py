import sys, re

path = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else "finn_design_i."

scope = []
id2keys = {}
with open(path, errors="replace") as f:
    for line in f:
        line = line.strip()
        if line.startswith("$scope"):
            scope.append(line.split()[2])
        elif line.startswith("$upscope"):
            if scope:
                scope.pop()
        elif line.startswith("$var"):
            p = line.split()
            name = ".".join(scope + [p[4]])
            low = name.lower()
            if prefix.lower() not in low:
                continue
            m = re.match(r"(.*)_(tvalid|tready)$", low)
            if not m:
                continue
            base = name[: len(m.group(1))]
            id2keys.setdefault(p[3], []).append((base, m.group(2)))
        elif line.startswith("$enddefinitions"):
            break
    last = {}
    last_t = {}
    t = 0
    for line in f:
        if line[0] == "#":
            t = int(line[1:])
            continue
        c = line[0]
        if c in "01xzXZ":
            i = line[1:].strip()
            if i in id2keys:
                last[i] = c
                last_t[i] = t

state = {}
for i, keys in id2keys.items():
    for base, kind in keys:
        state.setdefault(base, {})[kind] = (last.get(i, "?"), last_t.get(i, 0))

short = prefix
rows = []
for base, d in state.items():
    v = d.get("tvalid", ("?", 0))
    r = d.get("tready", ("?", 0))
    tag = {("1", "0"): "BLOCKED(data,no ready)", ("0", "1"): "STARVED(ready,no data)",
           ("0", "0"): "neither", ("1", "1"): "both1"}.get((v[0], r[0]), "?")
    rows.append((max(v[1], r[1]), base.split(short)[-1], v[0], r[0], tag))
rows.sort()
for lt, b, v, r, tag in rows:
    print(f"{lt:7d} V={v} R={r} {tag:24s} {b}")
