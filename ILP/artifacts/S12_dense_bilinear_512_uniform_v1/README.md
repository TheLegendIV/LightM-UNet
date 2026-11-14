# S12_dense_bilinear_512_uniform_v1

`layer_bits_SITES_uniform_int{4,6,8}.json`: uniform per-quantizer-SITE bits (85 weight sites, 110 activation sites, every one 4 / 6 / 8) of the S12 dense net with the **bilinear decoder**
(`decoder_type="upsample_conv"`, U4 widths (4, 16, 32, 16, 4), bottlenecks (4, 8, 8, 2, 1), `dense_dilation`) for the **512x512** FP32 net `nnUNetTrainerENet_12_dense_relu_warmstart150ep`
(Dataset509_ARCADE_1x1_4c). The site set is resolution-independent, so these are the key sets of `S12_dense_bilinear_256_uniform_int6_v1` with the values replaced (the int6 file is identical to it).
The frozen depthwise 3x3 tent conv of `LayerQuantEnetFINN` has no site, so it is not affected by the bit width (INT8 weights, fixed).

Used by `compression/slurm/qat_12_dense_relu_warmstart150ep_bilinear_uniform_int468_ft15ep_array.job` (array 0-2 = INT4 / INT6 / INT8).
