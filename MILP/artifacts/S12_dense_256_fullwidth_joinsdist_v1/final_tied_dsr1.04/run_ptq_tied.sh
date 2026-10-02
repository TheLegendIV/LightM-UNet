#!/bin/bash
# Tied-bits PTQ: expand sites with the tie, calibrate on ALL 1200 preprocessed images (odd shapes reflect-padded to x8),
# then test-split evaluation (collect_results.py, checkpoint_best only). Run in lightmunet_dev from /workspace/LightM-UNet.
set -e
cd /workspace/LightM-UNet
export nnUNet_raw=/workspace/LightM-UNet/data/nnUNet_raw nnUNet_preprocessed=/workspace/LightM-UNet/data/nnUNet_preprocessed nnUNet_results=/workspace/LightM-UNet/data/nnUNet_results
export ENET_RESIDUAL_ADD_FOLLOWS_BITS=1
D=MILP/artifacts/S12_dense_256_fullwidth_joinsdist_v1/final_tied_dsr1.04
NAME=nnUNetTrainerLayerQuantENet_12_dense_relu_nearest_conv_upsample_256_joint_alpha1.0_candidatebits468_lut70_bram40_dsp90_fps100_dsr1.04_jointsdist_tied_calibrated_all1200pad8
python3 MILP/expand_layer_bits.py --config config_12_dense_relu_nearest_conv_upsample_256 --tie-residual-bits --ilp-result $D/layer_bits_folding_final.json --out-file $D/layer_bits_SITES_final.json
python3 compression/post-quantization/calibrate_12_dense_relu_nearest_conv_upsample_256_perlayer.py --layer-bits-file $D/layer_bits_SITES_final.json --out-net-name $NAME --n-calibration-images 1200 --pad-to-multiple 8
export ENET_LAYER_BITS_FILE=/workspace/LightM-UNet/$D/layer_bits_SITES_final.json
rm -rf data/nnUNet_raw/Dataset510_ARCADE_256_4c/labelsPr_$NAME
python3 compression/collect_results.py --net-name $NAME --dataset-name Dataset510_ARCADE_256_4c --dataset-id 510 --channels 4,16,32,16,4 --bottlenecks 4,8,8,2,1 --decoder-type nearest_conv_upsample --use-asymmetric 0 --context-pattern dense_dilation_half --separable-dilated 0 --quant-bits 32 --trainer-class nnUNetTrainerLayerQuantENet_12_dense_relu_nearest_conv_upsample_256_perlayer --stage ptq_calibrated_tied_dsr1.04_jointsdist --input-hw 256 256 --checkpoint-name checkpoint_best.pth
echo PTQ_TIED_DONE
