"""Track CUMULATIVE successful-transfer counts (valid&ready both=1) over time for a specific
set of handshakes, to see whether an input/output rate mismatch opens up SUDDENLY (one big
latency cliff) or DRIFTS open gradually (small persistent throughput imbalance accumulating).

Usage: python3 vcd_timeprofile.py <trace.vcd> <substr1> [<substr2> ...]
Prints, for each matched tvalid/tready pair, a cumulative-transfer-count curve sampled at
even time intervals across the whole trace.
"""
import sys

def main():
    vcd_path = sys.argv[1]
    substrs = [s.lower() for s in sys.argv[2:]]

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
                if vid not in id_to_name:
                    id_to_name[vid] = full_name
            elif line.startswith("$enddefinitions"):
                break

        # find candidate vid's: exact leaf interface signals only (shortest matching name),
        # to avoid double-counting aliases
        wanted = {}  # vid -> (handshake_key, 'valid'|'ready')
        seen_keys = set()
        for vid, name in id_to_name.items():
            low = name.lower()
            if not any(s in low for s in substrs):
                continue
            if low.endswith("_tvalid"):
                kind, suflen = "valid", len("_tvalid")
            elif low.endswith(".tvalid"):
                kind, suflen = "valid", len(".tvalid")
            elif low.endswith("_tready"):
                kind, suflen = "ready", len("_tready")
            elif low.endswith(".tready"):
                kind, suflen = "ready", len(".tready")
            else:
                continue
            # strip only the tvalid/tready suffix so distinct ports (in0/in1/out0/out1) stay distinct
            key = name[:-suflen]
            dedup = (key, kind)
            if dedup in seen_keys:
                continue
            seen_keys.add(dedup)
            wanted[vid] = (key, kind)

        print(f"Tracking {len(wanted)} signals:")
        for vid, (key, kind) in wanted.items():
            print(f"  {kind:6s} {key} (vid={vid})")

        cur_val = {}  # vid -> last value
        transfer_count = {}  # key -> count
        events = []  # (time, key) whenever a transfer completes (valid=1 & ready=1 observed together)
        cur_time = 0
        total_lines = 0
        for line in f:
            total_lines += 1
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
                if vid in wanted:
                    cur_val[vid] = c0
            elif c0 == "$":
                continue
            else:
                continue

        print(f"(scanned {total_lines} lines, this was a value-only pass; re-scanning for transfer events)")

    # second pass: need to track valid&ready together per handshake key at each time step.
    # Re-open and do a proper per-timestep pass.
    group = {}  # key -> {'valid': vid, 'ready': vid}
    for vid, (key, kind) in wanted.items():
        group.setdefault(key, {})[kind] = vid

    complete_pairs = {k: v for k, v in group.items() if "valid" in v and "ready" in v}
    print(f"\n{len(complete_pairs)} complete valid/ready pairs found:")
    for k in complete_pairs:
        print(f"  {k}")

    vid_of_interest = set()
    for v in complete_pairs.values():
        vid_of_interest.add(v["valid"])
        vid_of_interest.add(v["ready"])

    cur_val2 = {}
    transfer_times = {k: [] for k in complete_pairs}
    cur_time = 0
    with open(vcd_path, "r", errors="replace") as f:
        # skip header
        while True:
            line = f.readline()
            if not line or line.strip().startswith("$enddefinitions"):
                break
        prev_state = {}
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
                if vid in vid_of_interest:
                    cur_val2[vid] = c0
                    for key, v in complete_pairs.items():
                        if vid in (v["valid"], v["ready"]):
                            vv = cur_val2.get(v["valid"])
                            rv = cur_val2.get(v["ready"])
                            if vv == "1" and rv == "1" and prev_state.get(key) != (cur_time,):
                                transfer_times[key].append(cur_time)
                                prev_state[key] = (cur_time,)

    max_time = cur_time
    n_buckets = 20
    bucket_w = max(1, max_time // n_buckets)
    print(f"\nmax_time={max_time} half-cycles, bucket_width={bucket_w}")
    print(f"\n{'key':60s} " + " ".join(f"b{i:02d}" for i in range(n_buckets)) + "  TOTAL")
    for key, times in transfer_times.items():
        buckets = [0] * (n_buckets + 1)
        for t in times:
            b = min(t // bucket_w, n_buckets)
            buckets[b] += 1
        row = " ".join(f"{c:3d}" for c in buckets[:n_buckets])
        print(f"{key[-60:]:60s} {row}  {len(times)}")


if __name__ == "__main__":
    main()
