"""ENetCombo trainer with a Dice + CE + clDice + Betti-0 loss: clDice rewards
centerline overlap (encourages vessel connectivity end-to-end), Betti-0
separately penalizes excess connected components (directly discourages
fragmentation into disjoint same-class islands). See
nnunetv2/training/loss/cldice_betti.py for why the two terms are
complementary rather than redundant, and cldice.py / betti.py for each term's
own math and self-test.

New env vars on top of everything nnUNetTrainerENetCombo already reads
(COMBO_EXPERIMENT, COMBO_EPOCHS, COMBO_BATCH_SIZE, COMBO_LR, ...):
  COMBO_CLDICE_WEIGHT     -- weight on the clDice term, alongside weight_ce=1.0
                             and weight_dice=1.0 (the base trainer's
                             defaults). Default 0.5: nnUNetTrainerENetComboClDice
                             alone defaults this to 1.0, but here that budget
                             is split with Betti-0 below instead of doubling
                             the total topological loss magnitude.
  COMBO_BETTI_WEIGHT      -- weight on the Betti-0 term. Default 0.5, same
                             reasoning as COMBO_CLDICE_WEIGHT above.
  COMBO_CLDICE_ITERS      -- number of soft-skeletonization iterations. See
                             nnUNetTrainerENetComboClDice's docstring for the
                             full measured-vessel-width justification.
                             Default 12.
  COMBO_BETTI_PATCH_SIZE  -- side length of the random crop the Betti-0 term
                             is computed on each step. See
                             nnUNetTrainerENetComboBetti's docstring. Default
                             64. Set to 0 to disable cropping (full patch,
                             much slower).
  COMBO_BETTI_PROB_FLOOR  -- ignore pixels below this probability when
                             building the persistence diagram. Default 0.01.
"""
import os

import torch

from nnunetv2.training.loss.cldice_betti import DC_and_CE_and_clDice_and_Betti_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENetCombo import nnUNetTrainerENetCombo


class nnUNetTrainerENetComboClDiceBetti(nnUNetTrainerENetCombo):

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.cldice_weight = float(os.environ.get("COMBO_CLDICE_WEIGHT", "0.5"))
        self.cldice_num_iter = int(os.environ.get("COMBO_CLDICE_ITERS", "12"))
        self.betti_weight = float(os.environ.get("COMBO_BETTI_WEIGHT", "0.5"))
        patch_size = int(os.environ.get("COMBO_BETTI_PATCH_SIZE", "64"))
        self.betti_patch_size = None if patch_size <= 0 else patch_size
        self.betti_prob_floor = float(os.environ.get("COMBO_BETTI_PROB_FLOOR", "0.01"))

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError(
                "DC_and_CE_and_clDice_and_Betti_loss assumes a single label map target "
                "(ARCADE's case), not region-based (multi-hot) targets."
            )

        loss = DC_and_CE_and_clDice_and_Betti_loss(
            soft_dice_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp,
            },
            ce_kwargs={},
            cldice_kwargs={"num_iter": self.cldice_num_iter, "do_bg": False},
            betti_kwargs={
                "do_bg": False,
                "patch_size": self.betti_patch_size,
                "prob_floor": self.betti_prob_floor,
            },
            weight_ce=1.0,
            weight_dice=1.0,
            weight_cldice=self.cldice_weight,
            weight_betti=self.betti_weight,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        # enable_deep_supervision is False for this trainer lineage
        # (nnUNetTrainerLightMUNet -> nnUNetTrainerNoDeepSupervision), so
        # unlike nnUNetTrainer._build_loss there is no DeepSupervisionWrapper
        # branch to mirror here -- `loss` is used as-is on the single output.
        return loss
