import sys, re

path = sys.argv[1]
targets = [t.lower() for t in sys.argv[2:]]  # exact base names, e.g. fmpadding_pixel_hls_0.in0_v

scope = []
id2 = {}
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
            m = re.match(r"(.*)_(tvalid|tready)$", name.lower())
            if not m:
                continue
            for t in targets:
                if m.group(1).endswith("finn_design_i." + t):
                    id2.setdefault(p[3], []).append((t, m.group(2)))
        elif line.startswith("$enddefinitions"):
            break
    cur = {}
    t_now = 0
    segs = {t: [] for t in targets}  # (start, end) of both==1
    open_at = {t: None for t in targets}
    NB = 20
    max_t = 0
    state = {t: {"tvalid": "0", "tready": "0"} for t in targets}
    for line in f:
        if line[0] == "#":
            t_now = int(line[1:])
            max_t = t_now
            continue
        c = line[0]
        if c not in "01xzXZ":
            continue
        i = line[1:].strip()
        if i not in id2:
            continue
        for t, kind in id2[i]:
            state[t][kind] = c
            both = state[t]["tvalid"] == "1" and state[t]["tready"] == "1"
            if both and open_at[t] is None:
                open_at[t] = t_now
            elif not both and open_at[t] is not None:
                segs[t].append((open_at[t], t_now))
                open_at[t] = None
    for t in targets:
        if open_at[t] is not None:
            segs[t].append((open_at[t], max_t))

bw = max_t / 20
print("max_time", max_t, "bucket", bw)
for t in targets:
    buckets = [0] * 20
    for a, b in segs[t]:
        for h in range(a, b):
            buckets[min(19, int(h / bw))] += 1
    tot = sum(buckets) / 2
    print(f"{t:40s} transfer-cycles={tot:.0f}  buckets(half-cycles/2):", [round(x / 2) for x in buckets])
