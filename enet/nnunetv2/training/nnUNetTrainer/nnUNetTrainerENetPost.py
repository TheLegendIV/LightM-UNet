"""Trainer for ENetPost -- ENetOriginal's own architecture (nnunetv2.nets.ENet.ENet,
via nnUNetTrainerENet, NOT the ENetCombo family -- no attention gates/skip
variants, just the paper-faithful net with however many input channels the
dataset declares) trained on Dataset509_ARCADE_ENetPost: the original
Dataset501 grayscale image (channel 0) plus a second channel (channel 1)
carrying nnUNetTrainerSmallENet's own reconstructed vessel-probability map
(trained on Dataset507_ARCADE_refinement -- see
dataset-prep/prepare_arcade_509_enetpost.py for exactly how that channel was
built). Nothing architectural changes to add the channel -- ENet.__init__
already takes in_channels from num_input_channels
(nnUNetTrainerENet.build_network_architecture), so a 2-channel dataset just
works; ENetPost only exists as its own trainer to (a) name this specific
recipe distinctly, matching the rest of this repo's per-experiment trainer
convention (nnUNetTrainerENetCombo, nnUNetTrainerENetComboClDice, ...), and
(b) add the clDice loss term below.

_build_loss uses DC_and_CE_and_clDice_loss (Dice + CE + soft-clDice,
softmax/multi-class -- Dataset509 keeps Dataset501's original 4-class
{background, LAD, RCA, LCX} labels, unlike SmallENet's single-sigmoid
binary convention). weight_cldice defaults to 1.0 -- the same weight as
Dice and CE, not a token amount -- computed with do_bg=False (clDice scores
each of the 3 foreground classes -- LAD/RCA/LCX -- and averages; background
connectivity isn't a meaningful concept here, same convention
nnUNetTrainerENetComboClDice already uses). Tune with
ENETPOST_CLDICE_WEIGHT (0.0 reproduces plain Dice+CE) and
ENETPOST_CLDICE_ITERS (default 12 -- covers p99 vessel width with margin,
see cldice.py's soft_skeletonize docstring for the measurement this is
based on).

Everything else (ENET_LR, ENET_EPOCHS, ENET_BATCH_SIZE, ENET_CHANNELS,
ENET_OUTPUT_FOLDER, ENET_SKIP_FINAL_VALIDATION, ...) is inherited unchanged
from nnUNetTrainerENet -- ENetPost is a strict superset, not a fork.
"""
import os

import torch

from nnunetv2.training.loss.cldice import DC_and_CE_and_clDice_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import nnUNetTrainerENet


class nnUNetTrainerENetPost(nnUNetTrainerENet):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.cldice_weight = float(os.environ.get("ENETPOST_CLDICE_WEIGHT", "1.0"))
        self.cldice_num_iter = int(os.environ.get("ENETPOST_CLDICE_ITERS", "12"))

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
