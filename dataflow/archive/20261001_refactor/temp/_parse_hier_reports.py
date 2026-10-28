import re
from collections import defaultdict

def parse(fn, top_name):
    by_cat = defaultdict(lambda: {"ramb36": 0, "ramb18": 0, "uram": 0, "dsp": 0, "count": 0})
    with open(fn) as f:
        lines = f.readlines()
    for line in lines:
        if not line.startswith("|"):
            continue
        rest = line[1:]
        stripped = rest.lstrip(" ")
        indent = len(rest) - len(stripped)
        if indent != 5:
            continue
        if stripped.startswith("("):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 10:
            continue
        inst = cols[0]
        if not inst.startswith(top_name + "_"):
            continue
        name = inst[len(top_name) + 1:]
        m = re.match(r"^(.*)_(\d+)$", name)
        cat = m.group(1) if m else name
        try:
            ramb36 = int(cols[7])
            ramb18 = int(cols[8])
            uram = int(cols[9])
            dsp = int(cols[10])
        except ValueError:
            continue
        by_cat[cat]["ramb36"] += ramb36
        by_cat[cat]["ramb18"] += ramb18
        by_cat[cat]["uram"] += uram
        by_cat[cat]["dsp"] += dsp
        by_cat[cat]["count"] += 1
    return by_cat


for fn, top in [("c:/DEV/repos/LightM-UNet/hardware/temp/hier_p6_full.rpt", "GenericPartition_6"),
                ("c:/DEV/repos/LightM-UNet/hardware/temp/hier_p7_full.rpt", "GenericPartition_7")]:
    print(f"\n=== {top} ===")
    by_cat = parse(fn, top)
    tot_18equiv = sum(v["ramb36"] * 2 + v["ramb18"] for v in by_cat.values())
    tot_uram = sum(v["uram"] for v in by_cat.values())
    tot_dsp = sum(v["dsp"] for v in by_cat.values())
    for cat, v in sorted(by_cat.items(), key=lambda kv: -(kv[1]["ramb36"] * 2 + kv[1]["ramb18"])):
        equiv = v["ramb36"] * 2 + v["ramb18"]
        if equiv == 0 and v["uram"] == 0 and v["dsp"] == 0:
            continue
        print(f"  {cat}: count={v['count']} RAMB36={v['ramb36']} RAMB18={v['ramb18']} equiv18k={equiv} URAM={v['uram']} DSP={v['dsp']}")
    print(f"  TOTAL: equiv18k={tot_18equiv} uram={tot_uram} dsp={tot_dsp}")
