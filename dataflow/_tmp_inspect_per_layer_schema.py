import json
import pprint

d = json.load(open("layer_bits_folding_final_tied_dsr104.json"))["per_layer"]
pprint.pprint(d["up5.reduce.0"])
print("---")
pprint.pprint(d["up5.skip_resize_conv.0"])
