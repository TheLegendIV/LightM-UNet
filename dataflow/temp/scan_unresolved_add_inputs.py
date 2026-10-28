import json
import glob

base = (
    r"C:\DEV\repos\LightM-UNet\hardware\builds\S12_dense_256_u4_analytical_v1\outputs\int6_fps250_lat200_milpfold_8way_20261005_235400"
    r"\intermediate_models_listing_only\supported_op_partitions"
)

for path in sorted(glob.glob(base + r"\fifo_force_report_partition_*.json")):
    pidx = path.split("_")[-1].split(".")[0]
    with open(path) as f:
        report = json.load(f)
    for e in report:
        cn = e.get("consumer_role") or ""
        if cn.endswith(".add") and e.get("producer_role") is None:
            print(f"partition {pidx}: fifo={e['fifo']} producer={e['producer_node']} consumer_role={cn} "
                  f"forced_depth={e['forced_depth']} stock_depth={e['stock_depth']}")
