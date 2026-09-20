import glob
import os

base = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs"
for d in sorted(glob.glob(os.path.join(base, "*8way_full*"))):
    log = os.path.join(d, "build_dataflow.log")
    if not os.path.isfile(log):
        continue
    with open(log, errors="ignore") as f:
        text = f.read()
    if "step_combine_partitions" not in text:
        continue
    print("===", d, "===")
    lines = [l for l in text.splitlines() if "step_combine_partitions" in l or "Completed successfully" in l or "Traceback" in l]
    for l in lines[-6:]:
        print(" ", l)
