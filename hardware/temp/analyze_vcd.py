"""Analyze a verilator-generated trace.vcd to find handshake signals (tvalid/tready) that
are "stuck" (haven't toggled recently relative to the end of the trace) -- useful for
pinpointing exactly which AXI-stream interface in a stitched FINN design is deadlocked.

Usage: python3 analyze_vcd.py <path-to-trace.vcd> [--top N]
"""
import sys

def main():
    vcd_path = sys.argv[1]
    top_n = 80
    if "--top" in sys.argv:
        top_n = int(sys.argv[sys.argv.index("--top") + 1])

    id_to_name = {}
    scope_stack = []
    with open(vcd_path, "r", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("$scope"):
                parts = line.split()
                scope_stack.append(parts[2])
            elif line.startswith("$upscope"):
                if scope_stack:
                    scope_stack.pop()
            elif line.startswith("$var"):
                parts = line.split()
                vid = parts[3]
                sig_name = parts[4]
                full_name = ".".join(scope_stack + [sig_name])
                # multiple vars can alias same id (rare) -- keep first seen, note others
                if vid not in id_to_name:
                    id_to_name[vid] = full_name
                else:
                    id_to_name[vid] = id_to_name[vid] + "|" + full_name
            elif line.startswith("$enddefinitions"):
                break

        last_change = {}
        toggle_count = {}
        cur_time = 0
        for line in f:
            if not line:
                continue
            c0 = line[0]
            if c0 == "#":
                try:
                    cur_time = int(line[1:].strip())
                except ValueError:
                    pass
                continue
            if c0 in "01xzXZ":
                vid = line[1:].strip()
                val = c0
                prev = last_change.get(vid)
                if prev is None or prev[1] != val:
                    toggle_count[vid] = toggle_count.get(vid, 0) + 1
                last_change[vid] = (cur_time, val)
            elif c0 in "bBrR":
                sp = line.find(" ")
                if sp == -1:
                    continue
                val = line[1:sp]
                vid = line[sp + 1:].strip()
                prev = last_change.get(vid)
                if prev is None or prev[1] != val:
                    toggle_count[vid] = toggle_count.get(vid, 0) + 1
                last_change[vid] = (cur_time, val)
            elif c0 == "$":
                continue

    max_time = cur_time
    print(f"max_time={max_time} (half-cycle units) total_signals={len(id_to_name)}")

    interesting = []
    for vid, name in id_to_name.items():
        low = name.lower()
        if "tvalid" in low or "tready" in low:
            t, v = last_change.get(vid, (0, "?"))
            n_tog = toggle_count.get(vid, 0)
            interesting.append((name, t, v, n_tog))

    # sort: signals that stopped changing EARLIEST (longest stuck) first
    interesting.sort(key=lambda row: row[1])

    print(f"\n=== Top {top_n} most-stuck tvalid/tready signals (sorted by last-change time, earliest=stuck longest) ===")
    print(f"{'cycles_since_change':>20}  {'final':>5}  {'#toggles':>8}  name")
    for name, t, v, n_tog in interesting[:top_n]:
        print(f"{max_time - t:20d}  {v:>5}  {n_tog:8d}  {name}")

    # Pair up tvalid/tready sharing a common prefix and flag stuck handshakes
    print("\n=== Candidate stuck handshakes (tvalid=1 held, tready=0 held, both long-unchanged) ===")
    by_prefix = {}
    for name, t, v, n_tog in interesting:
        low = name.lower()
        if "tvalid" in low:
            prefix = low.split("tvalid")[0]
            by_prefix.setdefault(prefix, {})["valid"] = (name, t, v, n_tog)
        elif "tready" in low:
            prefix = low.split("tready")[0]
            by_prefix.setdefault(prefix, {})["ready"] = (name, t, v, n_tog)

    candidates = []
    for prefix, d in by_prefix.items():
        if "valid" in d and "ready" in d:
            vname, vt, vv, vtog = d["valid"]
            rname, rt, rv, rtog = d["ready"]
            if vv == "1" and rv == "0":
                stuck_since = min(max_time - vt, max_time - rt)
                candidates.append((stuck_since, vname, rname, vt, rt, vtog, rtog))
    candidates.sort(key=lambda r: -r[0])
    for stuck_since, vname, rname, vt, rt, vtog, rtog in candidates[:top_n]:
        print(f"valid=1(toggled {vtog}x) & ready=0(toggled {rtog}x), stable for >= {stuck_since} half-cycles:")
        print(f"    VALID: {vname}")
        print(f"    READY: {rname}")


if __name__ == "__main__":
    main()
