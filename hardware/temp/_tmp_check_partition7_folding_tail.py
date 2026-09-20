import json
with open("/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_20260918_144349/hawq_folding_config_partition7.json") as f:
    d = json.load(f)
for k, v in list(d.items())[-6:]:
    print(k, v)
