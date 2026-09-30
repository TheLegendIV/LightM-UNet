set -e
cd /workspace/LightM-UNet
export ENET_LAYER_BITS_FILE=/workspace/LightM-UNet/MILP/artifacts/S12_dense_arms_bc_v1/lex_dsr2_fps305/layer_bits_SITES_lex_dsr2_fps305.json
export nnUNet_raw=/workspace/LightM-UNet/data/nnUNet_raw nnUNet_preprocessed=/workspace/LightM-UNet/data/nnUNet_preprocessed nnUNet_results=/workspace/LightM-UNet/data/nnUNet_results
TRAINER=nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_upsample_perlayer
RUN=12_dense_relu_nearest_upsample_wm_lex_dsr2_fps305_perlayer_candidatebits468_ft15ep
NET=${TRAINER}_${RUN}
ARGS=(--net-name $NET --dataset-name Dataset509_ARCADE_1x1_4c --dataset-id 509 --channels 4,16,32,16,4 --bottlenecks 4,8,8,2,1 --decoder-type nearest_upsample --use-asymmetric 0 --context-pattern dense_dilation --separable-dilated 0 --quant-bits 32 --trainer-class $TRAINER)
PRED=$nnUNet_raw/Dataset509_ARCADE_1x1_4c/labelsPr_$NET
rm -rf $PRED
python compression/collect_results.py "${ARGS[@]}" --stage $RUN --checkpoint-name checkpoint_best.pth
for EP in 5 10 15; do
  rm -rf $PRED
  python compression/collect_results.py "${ARGS[@]}" --stage ${RUN}_epoch$EP --config-name ${NET}_epoch$EP --checkpoint-name checkpoint_epoch$EP.pth
done
