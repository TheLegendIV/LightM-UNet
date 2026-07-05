"""Soft clDice loss (Shit et al., "clDice -- A Novel Topology-Preserving Loss
Function for Tubular Structure Segmentation", CVPR 2021) and a Dice+CE+clDice
compound loss to plug into an nnU-Net trainer.

Dice scores area overlap: a prediction that's fully right except for a few
missing pixels along one thin vessel loses almost no Dice, even though those
few pixels disconnect the whole vessel into two pieces. clDice instead scores
overlap between each mask's *skeleton* (centerline) and the other mask -- a
break in the skeleton is exactly what a connectivity failure looks like, so
it penalizes that failure directly, in a way Dice structurally cannot.

Skeletonization itself (e.g. skimage's) isn't differentiable, so this uses
the soft-skeletonization from the paper: iterated differentiable morphological
erosion/opening (approximated with min/max pooling) that converges to
something close to a true skeleton while staying end-to-end trainable.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Differentiable morphological erosion via min-pooling. img: (B, C, H, W)
    for 2D or (B, C, D, H, W) for 3D, values expected in [0, 1]."""
    if img.dim() == 4:
        p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        return torch.min(p1, p2)
    if img.dim() == 5:
        p1 = -F.max_pool3d(-img, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        p2 = -F.max_pool3d(-img, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0))
        p3 = -F.max_pool3d(-img, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)
    raise ValueError(f"soft_erode expects a 4D (2D data) or 5D (3D data) tensor, got {img.dim()}D")


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """Differentiable morphological dilation via max-pooling."""
    if img.dim() == 4:
        return F.max_pool2d(img, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    if img.dim() == 5:
        return F.max_pool3d(img, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    raise ValueError(f"soft_dilate expects a 4D (2D data) or 5D (3D data) tensor, got {img.dim()}D")


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, num_iter: int = 5) -> torch.Tensor:
    """Iterative differentiable skeletonization (Shit et al. 2021, Algorithm 1).
    img: soft mask in [0, 1] (e.g. a softmax probability channel). Larger
    num_iter reaches thinner skeletons but costs more compute; 3-5 is enough
    for vessels a few pixels wide (each iteration erodes by ~1px)."""
    img1 = soft_open(img)
    skeleton = F.relu(img - img1)
    for _ in range(num_iter):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


class SoftclDiceLoss(nn.Module):
    """Soft clDice loss, adapted for nnU-Net's multi-class one-hot targets:
    computed per foreground class (background excluded, matching this repo's
    do_bg=False Dice convention) and averaged across classes.

    Unlike the Dice losses in dice.py, this does not support loss_mask /
    ignore_label -- ARCADE doesn't use an ignore label, and skeletonizing a
    masked-out region doesn't have an obviously correct definition, so that
    case is left unimplemented rather than silently wrong.
    """

    def __init__(self, apply_nonlin: Callable = None, num_iter: int = 3, smooth: float = 1.0, do_bg: bool = False):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.num_iter = num_iter
        self.smooth = smooth
        self.do_bg = do_bg

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        net_output: (B, C, ...) raw logits (nonlin applied internally if apply_nonlin was given)
        target: (B, 1, ...) integer label map, or (B, C, ...) already one-hot
        """
        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        if net_output.ndim != target.ndim:
            target = target.view((target.shape[0], 1, *target.shape[1:]))

        if net_output.shape == target.shape:
            y_onehot = target.float()
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device, dtype=net_output.dtype)
            y_onehot.scatter_(1, target.long(), 1)

        if not self.do_bg:
            net_output = net_output[:, 1:]
            y_onehot = y_onehot[:, 1:]

        skel_pred = soft_skeletonize(net_output, self.num_iter)
        skel_true = soft_skeletonize(y_onehot, self.num_iter)

        reduce_dims = (0,) + tuple(range(2, net_output.ndim))  # keep the class dim
        tprec = (torch.sum(skel_pred * y_onehot, dim=reduce_dims) + self.smooth) / \
                (torch.sum(skel_pred, dim=reduce_dims) + self.smooth)
        tsens = (torch.sum(skel_true * net_output, dim=reduce_dims) + self.smooth) / \
                (torch.sum(skel_true, dim=reduce_dims) + self.smooth)

        cl_dice = 1.0 - (2.0 * tprec * tsens) / (tprec + tsens + 1e-8)
        return cl_dice.mean()


class DC_and_CE_and_clDice_loss(nn.Module):
    """Dice + CE + soft-clDice. The Dice+CE part is identical to
    compound_losses.DC_and_CE_loss (same kwargs, same ignore_label handling);
    clDice is added as a third weighted term to additionally reward
    topological connectivity, which Dice+CE alone don't score."""

    def __init__(
        self,
        soft_dice_kwargs: dict,
        ce_kwargs: dict,
        cldice_kwargs: dict,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        weight_cldice: float = 1.0,
        ignore_label: int | None = None,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cldice = weight_cldice
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.cldice = SoftclDiceLoss(apply_nonlin=softmax_helper_dim1, **cldice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one-hot encoded target variables (DC_and_CE_and_clDice_loss)"
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

        return self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_cldice * cldice_loss


if __name__ == "__main__":
    # Quick shape/gradient self-test -- run directly (`python cldice.py`) on a
    # machine with torch installed (this repo's conda env) before relying on
    # this in a real training job. No CUDA needed; runs fine on CPU.
    torch.manual_seed(0)
    batch, num_classes, height, width = 2, 4, 32, 32  # matches ARCADE: 0=BG,1=LAD,2=RCA,3=LCX

    logits = torch.randn(batch, num_classes, height, width, requires_grad=True)
    target = torch.randint(0, num_classes, (batch, 1, height, width))

    cldice_loss_fn = SoftclDiceLoss(apply_nonlin=softmax_helper_dim1, num_iter=3)
    loss = cldice_loss_fn(logits, target)
    loss.backward()
    print("SoftclDiceLoss value:", loss.item())
    print("logits.grad is not None:", logits.grad is not None)
    print("logits.grad has NaNs:", torch.isnan(logits.grad).any().item())

    combo_loss_fn = DC_and_CE_and_clDice_loss(
        soft_dice_kwargs={"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False},
        ce_kwargs={},
        cldice_kwargs={"num_iter": 3, "do_bg": False},
        weight_ce=1.0, weight_dice=1.0, weight_cldice=1.0,
    )
    logits2 = torch.randn(batch, num_classes, height, width, requires_grad=True)
    combo = combo_loss_fn(logits2, target)
    combo.backward()
    print("DC_and_CE_and_clDice_loss value:", combo.item())
    print("logits2.grad is not None:", logits2.grad is not None)
