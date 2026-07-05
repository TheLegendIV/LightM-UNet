"""Dice + CE + soft-clDice + Betti-0, combining cldice.py's connectivity term
(rewards centerline overlap) with betti.py's persistent-homology term
(penalizes excess connected components) on top of the standard Dice+CE
compound loss. See those two modules for each term's own math, motivation,
and self-test -- this module only wires them together as one weighted sum.

The two terms are complementary, not redundant: clDice can score well on a
prediction that's fragmented into several same-class islands, as long as each
island individually overlaps the true centerline well (see betti.py's module
docstring); Betti-0 is the term that directly counts and penalizes those
excess pieces.
"""
from __future__ import annotations

import torch
from torch import nn

from nnunetv2.training.loss.betti import Betti0Loss
from nnunetv2.training.loss.cldice import SoftclDiceLoss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class DC_and_CE_and_clDice_and_Betti_loss(nn.Module):
    """Dice + CE + soft-clDice + Betti-0. The Dice+CE part is identical to
    compound_losses.DC_and_CE_loss; clDice and Betti-0 are added as two more
    weighted terms, each independently toggleable via its own weight."""

    def __init__(
        self,
        soft_dice_kwargs: dict,
        ce_kwargs: dict,
        cldice_kwargs: dict,
        betti_kwargs: dict,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        weight_cldice: float = 0.5,
        weight_betti: float = 0.5,
        ignore_label: int | None = None,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cldice = weight_cldice
        self.weight_betti = weight_betti
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.cldice = SoftclDiceLoss(apply_nonlin=softmax_helper_dim1, **cldice_kwargs)
        self.betti = Betti0Loss(apply_nonlin=softmax_helper_dim1, **betti_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one-hot encoded target variables "
                "(DC_and_CE_and_clDice_and_Betti_loss)"
            )
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        cldice_loss = self.cldice(net_output, target_dice) if self.weight_cldice != 0 else 0
        betti_loss = self.betti(net_output, target_dice) if self.weight_betti != 0 else 0

        return (
            self.weight_ce * ce_loss
            + self.weight_dice * dc_loss
            + self.weight_cldice * cldice_loss
            + self.weight_betti * betti_loss
        )


if __name__ == "__main__":
    # Quick shape/gradient self-test -- mirrors cldice.py's and betti.py's own
    # __main__ blocks, checking the combined loss runs end-to-end and produces
    # finite gradients with all four terms active at once.
    torch.manual_seed(0)
    batch, num_classes, height, width = 2, 4, 32, 32  # matches ARCADE: 0=BG,1=LAD,2=RCA,3=LCX

    logits = torch.randn(batch, num_classes, height, width, requires_grad=True)
    target = torch.randint(0, num_classes, (batch, 1, height, width))

    loss_fn = DC_and_CE_and_clDice_and_Betti_loss(
        soft_dice_kwargs={"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False},
        ce_kwargs={},
        cldice_kwargs={"num_iter": 3, "do_bg": False},
        betti_kwargs={"patch_size": None, "do_bg": False},
        weight_ce=1.0, weight_dice=1.0, weight_cldice=0.5, weight_betti=0.5,
    )
    loss = loss_fn(logits, target)
    loss.backward()
    print("DC_and_CE_and_clDice_and_Betti_loss value:", loss.item())
    print("logits.grad is not None:", logits.grad is not None)
    print("logits.grad has NaNs:", torch.isnan(logits.grad).any().item())
