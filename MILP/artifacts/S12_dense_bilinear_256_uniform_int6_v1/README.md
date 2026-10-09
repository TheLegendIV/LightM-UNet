# S12_dense_bilinear_256_uniform_int6_v1

`layer_bits_SITES_uniform_int6.json`: uniform INT6 per-quantizer-SITE bits (85 weight sites, 110 activation sites, all 6) of the S12 dense net with the **bilinear decoder** (`decoder_type="upsample_conv"`,
U4 widths (4, 16, 32, 16, 4), bottlenecks (4, 8, 8, 2, 1), `dense_dilation`, 256x256). Generated from `LayerQuantENet.layer_names_for(...)`; the site set is the same as the nearest_upsample net's
(`S12_dense_nearest_upsample_256_uniform_int6_v1`): only the network class' `main_up` differs. The frozen depthwise 3x3 tent conv that realises the bilinear resize in `LayerQuantEnetFINN` has no site
(its INT8 weights are fixed in `_nearest_depthwise_bilinear_kernel`).

Pipeline (all three steps of `compression/slurm/qat_12_dense_relu_bilinear_upsample_256_uniform_int6_ft15ep.job` after the FP32 job has produced its checkpoint):

1. `compression/slurm/stage_12_dense_relu_bilinear_upsample_256.job` -> `nnUNetTrainerENet_12_dense_relu_bilinear_upsample_256` (FP32, Dataset510, trained from scratch).
2. `compression/post-quantization/calibrate_12_dense_relu_bilinear_upsample_256_perlayer.py --model-class finn` -> `nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_256_uniform_int6_calibrated`
   (the INT6 LayerQuantEnetFINN checkpoint; 1200 images, reflect-padded to x8).
3. `nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_256_perlayer` QAT, 15 epochs, warm-started from 2.; results row via `collect_results.py` on `checkpoint_best.pth`.

Wiring checked with a random-weight FP32 stand-in (not a result): the FINN-class calibration runs, the checkpoint carries the trainer name above and the frozen tent kernels
(`up4.main_up.1.weight`, `up5.main_up.1.weight`, shape [C, 1, 3, 3], kernel sum 1), and `LayerQuantEnetFINN.from_pretrained` reloads it with 0 shape mismatches.
