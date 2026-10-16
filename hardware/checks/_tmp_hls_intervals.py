import glob
import os
import re
import sys

root = sys.argv[1]
for rpt in sorted(glob.glob(f"{root}/code_gen_ipgen_*/project_*/sol1/syn/report/*_csynth.rpt")):
    name = os.path.basename(rpt).replace("_csynth.rpt", "")
    if "_Pipeline_" in name or "_Loop" in name:
        continue
    txt = open(rpt).read()
    m = re.search(r"\+ Latency:.*?\n\s+\|\s+(\d+)\|\s+(\d+)\|.*?\|\s+(\d+)\|\s+(\d+)\|\s+(\S+)\|", txt, re.S)
    if m:
        print(f"{name:70s} latency {m.group(1)}-{m.group(2)} interval {m.group(3)}-{m.group(4)} type={m.group(5)}")
