"""Phase 2: parse the full VCD, track final (frozen) value + last-change time
of ap_clk and every node output tvalid/tready signal, then print a
graph-ordered table so we can see exactly where the "backed up" (tvalid=1,
tready=0) region turns into the "starved" (tvalid=0) region -- that boundary
is the deadlock root cause.
"""
import sys
import re
import json

VCD_PATH = sys.argv[1]
NODE_LIST_JSON = sys.argv[2]  # produced by dump_partition1_nodes.py (modified) -- ordered list of node names

# ---- Phase A: parse header, build code -> signal name map (only signals we care about) ----
sig_re = re.compile(r"^(.*?)_(out\d?)_V_(TVALID|TREADY)$")

code_to_sig = {}  # code -> canonical signal name (first occurrence wins)
scope_stack = []
target_scope_prefix = None

with open(VCD_PATH, "r", errors="ignore") as f:
    for line in f:
        line = line.rstrip("\n")
        s = line.strip()
        if s.startswith("$scope"):
            parts = s.split()
            scope_stack.append(parts[2])
        elif s.startswith("$upscope"):
            scope_stack.pop()
        elif s.startswith("$var"):
            parts = s.split()
            code = parts[3]
            name = " ".join(parts[4:-1])
            cur_scope = "/".join(scope_stack)
            if cur_scope.endswith("GenericPartition_1_i"):
                if code not in code_to_sig:
                    code_to_sig[code] = name
            elif name in ("ap_clk",):
                if "ap_clk" not in code_to_sig.values():
                    code_to_sig[code] = "ap_clk"
            elif cur_scope == "TOP" and name in ("s_axis_0_tvalid", "s_axis_0_tready", "m_axis_0_tvalid", "m_axis_0_tready"):
                if code not in code_to_sig:
                    code_to_sig[code] = "TOP_" + name
        elif s.startswith("$enddefinitions"):
            break

print("Tracked signal codes:", len(code_to_sig), file=sys.stderr)

# ---- Phase B: scan value changes, track final value + last-change time per code ----
final_value = {}
last_change_time = {}
cur_time = 0
n_lines = 0
with open(VCD_PATH, "r", errors="ignore") as f:
    # skip header
    in_header = True
    for line in f:
        line = line.rstrip("\n")
        if in_header:
            if line.strip().startswith("$enddefinitions"):
                in_header = False
            continue
        n_lines += 1
        if not line:
            continue
        c0 = line[0]
        if c0 == "#":
            cur_time = int(line[1:])
        elif c0 in "01xzXZ":
            code = line[1:]
            if code in code_to_sig:
                val = c0
                if final_value.get(code) != val:
                    last_change_time[code] = cur_time
                final_value[code] = val
        elif c0 == "b":
            # multibit, space then code -- not used for our 1-bit signals, skip
            pass

print("Final sim time:", cur_time, "lines scanned:", n_lines, file=sys.stderr)

# ---- Build name -> (final_value, last_change_time) ----
name_info = {}
for code, name in code_to_sig.items():
    name_info[name] = (final_value.get(code, "?"), last_change_time.get(code, -1))

print("TOP s_axis_0_tvalid:", name_info.get("TOP_s_axis_0_tvalid"))
print("TOP s_axis_0_tready:", name_info.get("TOP_s_axis_0_tready"))
print("TOP m_axis_0_tvalid:", name_info.get("TOP_m_axis_0_tvalid"))
print("TOP m_axis_0_tready:", name_info.get("TOP_m_axis_0_tready"))
print("ap_clk final:", name_info.get("ap_clk"))
print("final sim time:", cur_time)
print()

with open(NODE_LIST_JSON, "r") as f:
    nodes = json.load(f)

print("%-4s %-45s %-10s %-22s %-22s" % ("idx", "node_name", "op_type", "out0 (tvalid,tready,t)", "out1 (tvalid,tready,t)"))
for idx, n in enumerate(nodes):
    name = n["name"]
    op_type = n["op_type"]
    out0_tvalid = name_info.get(name + "_out_V_TVALID") or name_info.get(name + "_out0_V_TVALID")
    out0_tready = name_info.get(name + "_out_V_TREADY") or name_info.get(name + "_out0_V_TREADY")
    out1_tvalid = name_info.get(name + "_out1_V_TVALID")
    out1_tready = name_info.get(name + "_out1_V_TREADY")

    def fmt(v):
        if v is None:
            return "-"
        return "v=%s@%s" % (v[0], v[1])

    out0_str = "tv:%s tr:%s" % (fmt(out0_tvalid), fmt(out0_tready))
    out1_str = "tv:%s tr:%s" % (fmt(out1_tvalid), fmt(out1_tready)) if (out1_tvalid or out1_tready) else ""
    print("%-4d %-45s %-10s %-35s %-35s" % (idx, name, op_type, out0_str, out1_str))
