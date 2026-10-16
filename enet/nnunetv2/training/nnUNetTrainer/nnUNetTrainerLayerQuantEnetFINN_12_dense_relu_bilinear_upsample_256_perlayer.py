"""LEGACY NAME of nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_perlayer (the `_256` was only the first resolution the trainer was written for; the class is resolution-independent).

Kept ONLY so that existing checkpoints (their embedded `trainer_name`) and result folders `nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_256_perlayer_*__nnUNetPlans__2d`
(Dataset509 512x512 and Dataset510 256x256 runs) still resolve with nnUNetv2_predict/--c. Do not use it for new runs: pass `-tr nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_perlayer`.
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_perlayer import (
    nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_perlayer,
)


class nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_256_perlayer(nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_perlayer):
    pass
