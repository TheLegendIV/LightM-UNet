import json

path = (
    r"C:\DEV\repos\LightM-UNet\hardware\builds\S12_dense_256_u4_analytical_v1\outputs\int6_fps250_lat200_milpfold_8way_20261005_235400"
    r"\intermediate_models_listing_only\supported_op_partitions\fifo_force_report_partition_1.json"
)
with open(path) as f:
    report = json.load(f)

print("total entries:", len(report))
for e in report:
    if e["fifo"] in ("StreamingFIFO_rtl_13", "StreamingFIFO_rtl_20", "StreamingFIFO_rtl_0", "StreamingFIFO_rtl_1",
                      "StreamingFIFO_rtl_2", "StreamingFIFO_rtl_26", "StreamingFIFO_rtl_44", "StreamingFIFO_rtl_62", "StreamingFIFO_rtl_80"):
        print(json.dumps(e, indent=2))

# also print a summary of matched vs unmatched
n_matched = sum(1 for e in report if e.get("forced_depth") is not None)
print(f"\nmatched {n_matched}/{len(report)}")
print("\nfirst 10 entries overall (graph order as stored):")
for e in report[:10]:
    print(json.dumps(e))
