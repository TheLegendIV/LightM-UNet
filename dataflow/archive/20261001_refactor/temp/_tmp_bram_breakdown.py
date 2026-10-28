import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from collections import defaultdict

OUT = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_nearest_conv_upsample_256_w8_16_v4_trained_rtl_mvau_8way_full_256x256_20260928_043156"

for pidx in (6, 7):
    fn = OUT + f"/intermediate_models/supported_op_partitions/partition_{pidx}.onnx"
    m = ModelWrapper(fn)
    by_type = defaultdict(lambda: {"bram18": 0.0, "uram": 0.0, "count": 0, "nodes": []})
    for n in m.graph.node:
        try:
            inst = getCustomOp(n)
        except Exception:
            continue
        b18 = 0.0
        ur = 0.0
        try:
            b18 = inst.bram_estimation()
        except Exception:
            pass
        try:
            ur = inst.uram_estimation()
        except Exception:
            pass
        cat = by_type[n.op_type]
        cat["bram18"] += b18
        cat["uram"] += ur
        cat["count"] += 1
        if b18 or ur:
            extra = {}
            for attr_name in ("SIMD", "PE", "Depth", "ram_style", "depth"):
                try:
                    extra[attr_name] = inst.get_nodeattr(attr_name)
                except Exception:
                    pass
            cat["nodes"].append((n.name, b18, ur, extra))

    print(f"\n=== partition_{pidx} ===")
    tot_b18 = sum(v["bram18"] for v in by_type.values())
    tot_ur = sum(v["uram"] for v in by_type.values())
    for op_type, v in sorted(by_type.items(), key=lambda kv: -kv[1]["bram18"]):
        if v["bram18"] == 0 and v["uram"] == 0:
            continue
        print(f"  {op_type}: count={v['count']} sum_bram18={v['bram18']:.0f} sum_uram={v['uram']:.0f}")
    print(f"  TOTAL (FINN native estimate): bram18={tot_b18:.0f} uram={tot_ur:.0f}")

    # show the single biggest BRAM-consuming nodes regardless of type
    all_nodes = [n for v in by_type.values() for n in v["nodes"]]
    all_nodes.sort(key=lambda x: -(x[1] + x[2]))
    print("  top individual nodes (name, bram18, uram, extra attrs):")
    for name, b18, ur, extra in all_nodes[:8]:
        print(f"    {name}: bram18={b18:.0f} uram={ur:.0f} {extra}")
