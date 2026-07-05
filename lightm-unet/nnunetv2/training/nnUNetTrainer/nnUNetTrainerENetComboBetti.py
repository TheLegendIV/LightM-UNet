"""ENetCombo trainer with a Dice + CE + Betti-0 loss, for directly
penalizing disconnected/fragmented predictions. See
nnunetv2/training/loss/betti.py for the persistent-homology math and its
self-test.

New env vars on top of everything nnUNetTrainerENetCombo already reads
(COMBO_EXPERIMENT, COMBO_EPOCHS, COMBO_BATCH_SIZE, COMBO_LR, ...):
  COMBO_BETTI_WEIGHT      -- weight on the Betti-0 term, alongside
                             weight_ce=1.0 and weight_dice=1.0 (the base
                             trainer's defaults). 0.0 reproduces the plain
                             Dice+CE loss exactly. Default 1.0.
  COMBO_BETTI_PATCH_SIZE  -- side length of the random crop the Betti-0 term
                             is computed on each step (bounds its CPU cost
                             regardless of the real patch size -- see
                             betti.py's cost warning). Default 64. Set to 0
                             to disable cropping and run on the full patch
                             (much slower).
  COMBO_BETTI_PROB_FLOOR  -- ignore pixels below this probability when
                             building the persistence diagram. Default 0.01.
"""
import os

import torch

from nnunetv2.training.loss.betti import DC_and_CE_and_Betti_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENetCombo import nnUNetTrainerENetCombo


class nnUNetTrainerENetComboBetti(nnUNetTrainerENetCombo):

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.betti_weight = float(os.environ.get("COMBO_BETTI_WEIGHT", "1.0"))
        patch_size = int(os.environ.get("COMBO_BETTI_PATCH_SIZE", "64"))
        self.betti_patch_size = None if patch_size <= 0 else patch_size
        self.betti_prob_floor = float(os.environ.get("COMBO_BETTI_PROB_FLOOR", "0.01"))

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError(
                "DC_and_CE_and_Betti_loss assumes a single label map target (ARCADE's case), "
                "not region-based (multi-hot) targets."
            )

        loss = DC_and_CE_and_Betti_loss(
            soft_dice_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp,
            },
            ce_kwargs={},
            betti_kwargs={
                "do_bg": False,
                "patch_size": self.betti_patch_size,
                "prob_floor": self.betti_prob_floor,
            },
            weight_ce=1.0,
            weight_dice=1.0,
            weight_betti=self.betti_weight,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        # enable_deep_supervision is False for this trainer lineage
        # (nnUNetTrainerLightMUNet -> nnUNetTrainerNoDeepSupervision), so
        # unlike nnUNetTrainer._build_loss there is no DeepSupervisionWrapper
        # branch to mirror here -- `loss` is used as-is on the single output.
        return loss
