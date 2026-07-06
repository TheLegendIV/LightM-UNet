"""ENetCombo trainer with a Dice + CE + clDice + territory-consistency loss:
clDice rewards centerline overlap (encourages vessel connectivity end-to-end),
territory-consistency directly penalizes predicting RCA in an LAD/LCX-only
image or vice versa -- ARCADE's hard anatomical rule that a single angiogram
is always exclusively one coronary tree (see
nnunetv2/training/loss/territory.py for the full rationale, and why this is
safe where Betti-0 was not: territory-consistency reads the true territory
straight off ground truth at training time, so unlike Betti-0 (or the
post-processing enforce_territory heuristic, which must guess) there is
nothing to guess and nothing that was measured to collapse training).

New env vars on top of everything nnUNetTrainerENetCombo already reads
(COMBO_EXPERIMENT, COMBO_EPOCHS, COMBO_BATCH_SIZE, COMBO_LR, ...):
  COMBO_CLDICE_WEIGHT     -- weight on the clDice term, alongside weight_ce=1.0
                             and weight_dice=1.0 (the base trainer's
                             defaults). Default 0.5, splitting the topology
                             loss budget with territory-consistency below
                             rather than doubling the total auxiliary loss
                             magnitude.
  COMBO_TERRITORY_WEIGHT  -- weight on the territory-consistency term.
                             Default 0.5, same reasoning as
                             COMBO_CLDICE_WEIGHT above.
  COMBO_TERRITORY_MARGIN  -- probability threshold below which wrong-territory
                             mass is ignored entirely (zero loss, zero
                             gradient -- see TerritoryLoss's docstring for why
                             this makes the term go fully silent once the
                             network already respects territory, instead of
                             endlessly pushing wrong-territory probability
                             toward zero). Default 0.05.
  COMBO_TERRITORY_POWER   -- exponent applied to the above-margin excess.
                             >1 makes a large violation cost much more than
                             one that just barely crosses the margin.
                             Default 2.0.
  COMBO_CLDICE_ITERS      -- number of soft-skeletonization iterations. See
                             nnUNetTrainerENetComboClDice's docstring for the
                             full measured-vessel-width justification.
                             Default 12.
"""
import os

import torch

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.territory import DC_and_CE_and_clDice_and_Territory_loss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENetCombo import nnUNetTrainerENetCombo


class nnUNetTrainerENetComboClDiceTerritory(nnUNetTrainerENetCombo):

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.cldice_weight = float(os.environ.get("COMBO_CLDICE_WEIGHT", "0.5"))
        self.cldice_num_iter = int(os.environ.get("COMBO_CLDICE_ITERS", "12"))
        self.territory_weight = float(os.environ.get("COMBO_TERRITORY_WEIGHT", "0.5"))
        self.territory_margin = float(os.environ.get("COMBO_TERRITORY_MARGIN", "0.05"))
        self.territory_power = float(os.environ.get("COMBO_TERRITORY_POWER", "2.0"))

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError(
                "DC_and_CE_and_clDice_and_Territory_loss assumes a single label map target "
                "(ARCADE's case), not region-based (multi-hot) targets."
            )

        loss = DC_and_CE_and_clDice_and_Territory_loss(
            soft_dice_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp,
            },
            ce_kwargs={},
            cldice_kwargs={"num_iter": self.cldice_num_iter, "do_bg": False},
            territory_kwargs={"margin": self.territory_margin, "power": self.territory_power},
            weight_ce=1.0,
            weight_dice=1.0,
            weight_cldice=self.cldice_weight,
            weight_territory=self.territory_weight,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        # enable_deep_supervision is False for this trainer lineage
        # (nnUNetTrainerLightMUNet -> nnUNetTrainerNoDeepSupervision), so
        # unlike nnUNetTrainer._build_loss there is no DeepSupervisionWrapper
        # branch to mirror here -- `loss` is used as-is on the single output.
        return loss
