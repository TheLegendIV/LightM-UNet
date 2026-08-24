import json, re, collections

d = json.load(open(r"c:\DEV\repos\LightM-UNet\hardware\_tmp_estimate_layer_resources.json"))
totals = collections.defaultdict(collections.Counter)
grand = collections.Counter()
for k, v in d.items():
    m = re.match(r"(GenericPartition_\d+)_", k)
    part = m.group(1) if m else "UNKNOWN"
    for res in ("LUT", "BRAM_18K", "URAM", "DSP"):
        totals[part][res] += v.get(res, 0)
        grand[res] += v.get(res, 0)

for part in sorted(totals.keys(), key=lambda x: int(x.split("_")[1]) if x.split("_")[-1].isdigit() else 999):
    t = totals[part]
    print("%-20s LUT=%7d  BRAM_18K=%4d  URAM=%3d  DSP=%3d" % (part, t["LUT"], t["BRAM_18K"], t["URAM"], t["DSP"]))
print("-" * 60)
print("%-20s LUT=%7d  BRAM_18K=%4d  URAM=%3d  DSP=%3d" % ("TOTAL", grand["LUT"], grand["BRAM_18K"], grand["URAM"], grand["DSP"]))
