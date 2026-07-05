"""ENetCombo trainer with a Dice + CE + soft-clDice loss, for encouraging
vessel connectivity in addition to area overlap. See
nnunetv2/training/loss/cldice.py for why clDice targets connectivity in a way
Dice structurally cannot, and its module docstring / self-test for the loss
math itself.

New env vars on top of everything nnUNetTrainerENetCombo already reads
(COMBO_EXPERIMENT, COMBO_EPOCHS, COMBO_BATCH_SIZE, COMBO_LR, ...):
  COMBO_CLDICE_WEIGHT  -- weight on the clDice term, alongside weight_ce=1.0
                          and weight_dice=1.0 (the base trainer's defaults).
                          0.0 reproduces the plain Dice+CE loss exactly, which
                          is a useful sanity-check baseline point in a
                          Pareto sweep over this value. Default 1.0.
  COMBO_CLDICE_ITERS   -- number of soft-skeletonization iterations (higher
                          reaches thinner skeletons, costs more compute).
                          Default 3, matching ARCADE's thin (few-px) vessels.
"""
import os

import torch

from nnunetv2.training.loss.cldice import DC_and_CE_and_clDice_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENetCombo import nnUNetTrainerENetCombo


class nnUNetTrainerENetComboClDice(nnUNetTrainerENetCombo):

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.cldice_weight = float(os.environ.get("COMBO_CLDICE_WEIGHT", "1.0"))
        self.cldice_num_iter = int(os.environ.get("COMBO_CLDICE_ITERS", "3"))

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError(
                "DC_and_CE_and_clDice_loss assumes a single label map target (ARCADE's case), "
                "not region-based (multi-hot) targets."
            )

        loss = DC_and_CE_and_clDice_loss(
            soft_dice_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp,
            },
            ce_kwargs={},
            cldice_kwargs={"num_iter": self.cldice_num_iter, "do_bg": False},
            weight_ce=1.0,
            weight_dice=1.0,
            weight_cldice=self.cldice_weight,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        # enable_deep_supervision is False for this trainer lineage
        # (nnUNetTrainerLightMUNet -> nnUNetTrainerNoDeepSupervision), so
        # unlike nnUNetTrainer._build_loss there is no DeepSupervisionWrapper
        # branch to mirror here -- `loss` is used as-is on the single output.
        return loss
