import json

path = (
    "/workspace/LightM-UNet/data/nnUNet_results/Dataset510_ARCADE_256_4c/"
    "nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_conv_upsample_256_perlayer_"
    "12_dense_relu_nearest_conv_upsample_256_finn_joint_alpha1.0_perlayer_candidatebits468_"
    "forcedsp_lut50_bram50_dsp90_fps250_dsrate1.5_ft15ep__nnUNetPlans__2d/fold_0/debug.json"
)
with open(path) as f:
    d = json.load(f)
for k, v in d.items():
    ku = k.upper()
    if "ENET" in ku or "ENV" in ku or "TRAINER" in ku or "PRETRAIN" in ku or "PRUN" in ku:
        print(k, ":", v)
