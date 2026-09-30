#!/bin/bash
BD=/tmp/finn_dev_thelegendiv/vivado_stitch_proj_d9ohtyc0/finn_vivado_stitch_proj.srcs/sources_1/bd/GenericPartition_2/GenericPartition_2.bd
python3 - "$BD" <<'EOF'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)

def walk(obj, path=""):
    results = []
    if isinstance(obj, dict):
        if "FIFO_MEMORY_TYPE" in obj.get("parameters", {}) if isinstance(obj.get("parameters"), dict) else False:
            pass
        for k, v in obj.items():
            results.extend(walk(v, path + "/" + str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            results.extend(walk(v, path))
    return results

# Simpler: find all cells and their FIFO_MEMORY_TYPE param recursively
def find_fifo_cells(obj, path=""):
    found = []
    if isinstance(obj, dict):
        # a "cells" dict: keys are instance names, values are cell defs
        if "cells" in obj and isinstance(obj["cells"], dict):
            for cellname, celldef in obj["cells"].items():
                params = celldef.get("parameters", {})
                if "FIFO_MEMORY_TYPE" in params:
                    val = params["FIFO_MEMORY_TYPE"].get("value", "?")
                    found.append((path + "/" + cellname, val))
                found.extend(find_fifo_cells(celldef, path + "/" + cellname))
        for k, v in obj.items():
            if k != "cells":
                found.extend(find_fifo_cells(v, path))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(find_fifo_cells(v, path))
    return found

results = find_fifo_cells(data)
print(f"Found {len(results)} cells with FIFO_MEMORY_TYPE")
for path, val in results:
    print(path, "=", val)
EOF
