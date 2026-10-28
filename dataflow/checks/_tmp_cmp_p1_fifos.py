import json
import sys

new_p = sys.argv[1]
old_p = sys.argv[2]
for label, path in (("INT8", new_p), ("mixed68", old_p)):
    r = json.load(open(path))
    print(f"== {label}: {len(r)} FIFOs")
    for e in r:
        pn, cn = e.get("producer_node") or "", e.get("consumer_node") or ""
        interior_unforced = e["forced_depth"] is None and pn and cn
        near_add = "AddStreams" in cn or "AddStreams" in pn
        big = (e["forced_depth"] or 0) >= 500
        if interior_unforced or near_add or big:
            print(f"  {e['fifo']:22s} {pn[:42]:42s} -> {cn[:42]:42s} forced={e['forced_depth']} stock={e.get('stock_depth')} roles={e.get('producer_role')}=>{e.get('consumer_role')}")
