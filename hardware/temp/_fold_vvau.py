import json
D = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200_milpfold_8way_20261009_225105/"
for k in (5, 6):
    d = json.load(open(D + "hawq_folding_config_partition%d.json" % k))
    print("== P%d keys=%d" % (k, len(d)))
    for n, v in d.items():
        if any(s in n for s in ("VVAU", "ConvolutionInputGenerator_rtl_3", "FMPadding_rtl_3", "Upsample", "Defaults", "StreamingDataWidthConverter_hls", "ConvolutionInputGenerator_3")):
            print(" ", n, v)
