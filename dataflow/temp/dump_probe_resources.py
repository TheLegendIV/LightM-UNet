"""Dump hierarchical (per-node) resources + landed folding + FIFO depths
for 3 ratchet-ablation probe builds, and diff them, as CSVs."""
import csv
import json
import os
import re

ENET = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs"
OUT = "/tmp/probe_dump"
os.makedirs(OUT, exist_ok=True)

BUILDS = {
    "ratchet_ablation_finn_autofold": (
        "ratchet_ablation_finn_autofold_autofold_partition2_20261007_023106",
        "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/synth_out_of_context_3oa532qe",
    ),
    "analytical_25pct": (
        "analytical_25pct_milpfold_partition2_20261007_052841",
        "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/synth_out_of_context_kuny6og0",
    ),
    "analytical_25pct_finnfifo": (
        "analytical_25pct_finnfifo_milpfold_partition2_20261007_060247",
        "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/synth_out_of_context_e9t2ghx5",
    ),
}

PREFIX = "GenericPartition_2_"


def load_build(tag, subdir, synthdir):
    base = os.path.join(ENET, subdir)
    with open(os.path.join(base, "auto_folding_config.json")) as f:
        folding = json.load(f)
    folding.pop("Defaults", None)
    with open(os.path.join(base, "report", "estimate_layer_resources_hls.json")) as f:
        res = json.load(f)
    with open(os.path.join(base, "report", "ooc_synth_and_timing.json")) as f:
        synth_total = json.load(f)

    # FIFO depths: parse ".depth(N)" out of each plain (non "finn_design_"
    # prefixed) GenericPartition_2_StreamingFIFO_rtl_*.v wrapper file.
    fifo_depths = {}
    if os.path.isdir(synthdir):
        for fn in os.listdir(synthdir):
            m = re.match(r"GenericPartition_2_(StreamingFIFO_rtl_\d+)\.v$", fn)
            if not m:
                continue
            with open(os.path.join(synthdir, fn)) as f:
                text = f.read()
            dm = re.search(r"\.depth\((\d+)\)", text)
            if dm:
                fifo_depths[m.group(1)] = int(dm.group(1))

    # merge: one row per node (keyed by bare node name, no GenericPartition_2_ prefix)
    all_nodes = set(folding) | {k[len(PREFIX):] for k in res if k.startswith(PREFIX)} | set(fifo_depths)
    rows = {}
    for node in all_nodes:
        fold = folding.get(node, {})
        r = res.get(PREFIX + node, {})
        rows[node] = {
            "PE": fold.get("PE", ""),
            "SIMD": fold.get("SIMD", ""),
            "ram_style": fold.get("ram_style", ""),
            "resType": fold.get("resType", ""),
            "depth_trigger_bram": fold.get("depth_trigger_bram", ""),
            "depth_trigger_uram": fold.get("depth_trigger_uram", ""),
            "fifo_depth": fifo_depths.get(node, ""),
            "LUT": r.get("LUT", ""),
            "FF": r.get("FF", ""),
            "BRAM_18K": r.get("BRAM_18K", ""),
            "DSP48E": r.get("DSP48E", ""),
            "URAM": r.get("URAM", ""),
        }
    return rows, synth_total


data = {}
totals = {}
for tag, (subdir, synthdir) in BUILDS.items():
    data[tag], totals[tag] = load_build(tag, subdir, synthdir)
    print(f"{tag}: {len(data[tag])} nodes, synth totals: {totals[tag]}")

fields = ["node", "PE", "SIMD", "ram_style", "resType", "depth_trigger_bram",
          "depth_trigger_uram", "fifo_depth", "LUT", "FF", "BRAM_18K", "DSP48E", "URAM"]

# per-build CSV
for tag, rows in data.items():
    path = os.path.join(OUT, f"{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for node in sorted(rows, key=lambda n: (n.rsplit("_", 1)[0], int(n.rsplit("_", 1)[1]) if n.rsplit("_", 1)[1].isdigit() else 0)):
            w.writerow({"node": node, **rows[node]})
    print(f"wrote {path}")

# synth totals CSV
with open(os.path.join(OUT, "synth_totals.csv"), "w", newline="") as f:
    keys = ["LUT", "LUTRAM", "FF", "DSP", "BRAM", "BRAM_18K", "BRAM_36K", "URAM", "Carry",
            "WNS", "fmax_mhz", "estimated_throughput_fps"]
    w = csv.writer(f)
    w.writerow(["tag"] + keys)
    for tag, t in totals.items():
        w.writerow([tag] + [t.get(k, "") for k in keys])
print(f"wrote {os.path.join(OUT, 'synth_totals.csv')}")

# folding diff: analytical_25pct vs analytical_25pct_finnfifo
a, b = data["analytical_25pct"], data["analytical_25pct_finnfifo"]
all_nodes = sorted(set(a) | set(b))
diff_rows = []
identical = True
for node in all_nodes:
    ra, rb = a.get(node, {}), b.get(node, {})
    fold_a = (ra.get("PE"), ra.get("SIMD"))
    fold_b = (rb.get("PE"), rb.get("SIMD"))
    if fold_a != fold_b:
        identical = False
    diff_rows.append({
        "node": node,
        "PE_analytical_25pct": ra.get("PE", ""), "SIMD_analytical_25pct": ra.get("SIMD", ""),
        "PE_finnfifo": rb.get("PE", ""), "SIMD_finnfifo": rb.get("SIMD", ""),
        "fold_match": fold_a == fold_b,
        "fifo_depth_analytical_25pct": ra.get("fifo_depth", ""),
        "fifo_depth_finnfifo": rb.get("fifo_depth", ""),
        "fifo_depth_match": ra.get("fifo_depth", "") == rb.get("fifo_depth", ""),
    })
with open(os.path.join(OUT, "diff_analytical_vs_finnfifo.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(diff_rows[0].keys()))
    w.writeheader()
    w.writerows(diff_rows)
print(f"FOLDING IDENTICAL between analytical_25pct and analytical_25pct_finnfifo: {identical}")
fifo_mismatches = [r for r in diff_rows if not r["fifo_depth_match"]]
print(f"FIFO depth mismatches between the two analytical arms: {len(fifo_mismatches)}")

# 3-way resource/folding diff (all 3 builds)
all_nodes3 = sorted(set(data["ratchet_ablation_finn_autofold"]) | set(a) | set(b))
with open(os.path.join(OUT, "diff_3way.csv"), "w", newline="") as f:
    cols = ["node"]
    for tag in BUILDS:
        cols += [f"PE_{tag}", f"SIMD_{tag}", f"fifo_depth_{tag}", f"LUT_{tag}"]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for node in all_nodes3:
        row = {"node": node}
        for tag in BUILDS:
            r = data[tag].get(node, {})
            row[f"PE_{tag}"] = r.get("PE", "")
            row[f"SIMD_{tag}"] = r.get("SIMD", "")
            row[f"fifo_depth_{tag}"] = r.get("fifo_depth", "")
            row[f"LUT_{tag}"] = r.get("LUT", "")
        w.writerow(row)
print(f"wrote {os.path.join(OUT, 'diff_3way.csv')}")
