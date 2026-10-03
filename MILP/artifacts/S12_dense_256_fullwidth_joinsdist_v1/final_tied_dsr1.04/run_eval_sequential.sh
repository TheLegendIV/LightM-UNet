#!/bin/bash
# Sequential test-split evaluations of the tied-bits scheme (calibrated checkpoints already exist): FINN-class first
# (AGENTS.md "Model class rule"), then LayerQuantENet, then the FINN export cross-test. Run in lightmunet_dev.
cd /workspace/LightM-UNet
export nnUNet_raw=/workspace/LightM-UNet/data/nnUNet_raw nnUNet_preprocessed=/workspace/LightM-UNet/data/nnUNet_preprocessed nnUNet_results=/workspace/LightM-UNet/data/nnUNet_results
D=MILP/artifacts/S12_dense_256_fullwidth_joinsdist_v1/final_tied_dsr1.04
export ENET_LAYER_BITS_FILE=/workspace/LightM-UNet/$D/layer_bits_SITES_final.json
BASE=joint_alpha1.0_candidatebits468_lut70_bram40_dsp90_fps100_dsr1.04_jointsdist_tied_calibrated_all1200pad8
COMMON="--dataset-name Dataset510_ARCADE_256_4c --dataset-id 510 --channels 4,16,32,16,4 --bottlenecks 4,8,8,2,1 --decoder-type nearest_conv_upsample --use-asymmetric 0 --context-pattern dense_dilation_half --separable-dilated 0 --quant-bits 32 --input-hw 256 256 --checkpoint-name checkpoint_best.pth"
for CLS in finn layerquant; do
  if [ $CLS = finn ]; then
    N=nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_conv_upsample_256_$BASE
    T=nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_conv_upsample_256_perlayer; S=ptq_calibrated_tied_finn_dsr1.04_jointsdist
  else
    N=nnUNetTrainerLayerQuantENet_12_dense_relu_nearest_conv_upsample_256_$BASE
    T=nnUNetTrainerLayerQuantENet_12_dense_relu_nearest_conv_upsample_256_perlayer; S=ptq_calibrated_tied_dsr1.04_jointsdist
  fi
  CK=data/nnUNet_results/Dataset510_ARCADE_256_4c/${N}__nnUNetPlans__2d/fold_0
  [ -f $CK/checkpoint_best.pth ] || cp $CK/checkpoint_best.pth.txt $CK/checkpoint_best.pth   # a sync tool renames .pth -> .pth.txt
  rm -rf data/nnUNet_raw/Dataset510_ARCADE_256_4c/labelsPr_$N
  python3 compression/collect_results.py --net-name $N --trainer-class $T --stage $S $COMMON
  echo "EVAL_DONE $CLS"
done
python3 -u hardware/builds/12_dense_relu_nearest_conv_upsample_256_v3/finn_export_12_dense_relu_nearest_conv_upsample_256_v3_trained.py --skip-export
echo "ALL_SEQ_DONE"
