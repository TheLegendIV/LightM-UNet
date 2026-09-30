import json
d = json.load(open("MILP/artifacts/S12_dense_dsr_ablation_v1/dsr_off/layer_bits_folding_dsr_off.json"))
pl = d["per_layer"]
keys1 = ["down1.reduce.0", "down1.conv.0", "down1.expand.0",
         "regular1.0.reduce.0", "regular1.0.conv", "regular1.0.expand.0",
         "regular1.1.reduce.0", "regular1.1.conv", "regular1.1.expand.0",
         "regular1.2.reduce.0", "regular1.2.conv", "regular1.2.expand.0",
         "regular1.3.reduce.0", "regular1.3.conv", "regular1.3.expand.0"]
keys2 = ["stage2.0.reduce.0", "stage2.0.conv", "stage2.0.expand.0",
         "stage2.1.reduce.0", "stage2.1.conv", "stage2.1.expand.0",
         "stage2.2.reduce.0", "stage2.2.conv", "stage2.2.expand.0",
         "stage2.3.reduce.0", "stage2.3.conv", "stage2.3.expand.0",
         "stage2.4.reduce.0", "stage2.4.conv", "stage2.4.expand.0"]
for k1, k2 in zip(keys1, keys2):
    e1, e2 = pl[k1], pl[k2]
    print(f"{k1:22s} pe={e1['pe']} simd={e1['simd']} cyc={e1['mvu_cycles']}  ||  "
          f"{k2:22s} pe={e2['pe']} simd={e2['simd']} cyc={e2['mvu_cycles']}")
