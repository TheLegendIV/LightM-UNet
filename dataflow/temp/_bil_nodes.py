import json, re
O = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
B = O + "S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200_milpfold_8way_20261009_225105/"
est = json.load(open(B + "report/estimate_layer_cycles.json"))
def short(k): return re.sub(r"^GenericPartition_(\d+)_GenericPartition_\d+_", r"P\1:", k)
for p in (4, 5, 6):
    print("=== P%d (non-FIFO nodes, in report order)" % p)
    for k, v in est.items():
        if k.startswith("GenericPartition_%d_" % p) and "StreamingFIFO" not in k:
            print("  ", short(k), v)
