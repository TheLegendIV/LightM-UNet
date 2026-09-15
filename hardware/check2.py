import json
with open("/home/thelegendiv/finn/notebooks/enet/quantEnet_12_dense_relu_warmstart150ep_alpha025_dummy_int8_conv_order.json") as f:
    all_names = json.load(f)
for i, e in enumerate(all_names):
    if "shortcut_proj" in e["logical_name"] or "main_up" in e["logical_name"]:
        print(i, e)
print("total entries:", len(all_names))