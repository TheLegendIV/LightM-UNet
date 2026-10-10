import json
d = json.load(open("/tmp/conv_order.json"))
for i, e in enumerate(d):
    if e["logical_name"].startswith(("down1", "down2", "regular1.0")):
        print(i, e["logical_name"].ljust(22), e["module_type"].ljust(14), e.get("weight_shape"))
