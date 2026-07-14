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


def soft_skeletonize(img: torch.Tensor, num_iter: int = 12) -> torch.Tensor:
    """Iterative differentiable skeletonization (Shit et al. 2021, Algorithm 1).
    img: soft mask in [0, 1] (e.g. a softmax probability channel).

    Each iteration erodes by ~1px per side, so a structure of width W needs
    ~W/2 iterations to fully converge to its 1px centerline -- iterating past
    that point is harmless (an already-converged region just stops
    contributing further, verified by direct simulation), so there's no
    downside to over-provisioning num_iter except compute cost. Measured on
    ARCADE's actual ground truth (via distance-transform width at skeleton
    pixels, all 300 test cases): median vessel width ~8-10px, p95 ~15-19px,
    p99 ~20-24px (RCA widest), max ~32px. num_iter=3 (this module's original
    default) only fully skeletonizes vessels up to ~8px -- below the median
    for every class -- so wider (mostly proximal) segments got an empty
    soft-skeleton and zero clDice signal. 12 covers p99 for all three classes
    with margin; raise toward ~16 to also cover the rare ~32px max-width
    segments, at proportionally more compute per training step."""
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

    def __init__(self, apply_nonlin: Callable = None, num_iter: int = 12, smooth: float = 1.0, do_bg: bool = False):
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


class DC_and_BCE_and_clDice_loss(nn.Module):
    """Dice + BCE + soft-clDice, for a single-logit sigmoid output (e.g.
    nnUNetTrainerSmallENet). The Dice+BCE part is identical to
    compound_losses.DC_and_BCE_loss (same kwargs, same target[:, -1]
    ignore-channel convention); clDice is added as a third weighted term.

    Unlike the softmax/CE version above (DC_and_CE_and_clDice_loss), there is
    no background channel to exclude -- the single output channel *is* the
    foreground -- so SoftclDiceLoss must be constructed with do_bg=True here.
    """

    def __init__(
        self,
        bce_kwargs: dict,
        soft_dice_kwargs: dict,
        cldice_kwargs: dict,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        weight_cldice: float = 1.0,
        use_ignore_label: bool = False,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        super().__init__()
        if use_ignore_label:
            bce_kwargs = {**bce_kwargs, "reduction": "none"}

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cldice = weight_cldice
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)
        self.cldice = SoftclDiceLoss(apply_nonlin=torch.sigmoid, **cldice_kwargs)

    def forward(
        self, net_output: torch.Tensor, target: torch.Tensor, pixel_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """pixel_weight: optional (B, 1, ...) elementwise BCE weight, same
        shape as target -- e.g. nnUNetTrainerSmallRefinementENet's gap-focused
        weighting (higher weight where the input predicted-mask channel
        disagrees with GT, so the loss cares about *changing* those pixels,
        not just reproducing the ones a trivial passthrough already gets
        right). Only scales the BCE term; Dice/clDice score global area
        overlap and skeleton topology respectively, not something a per-pixel
        weight map has an obviously correct way to modulate."""
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.weight_ce != 0:
            if pixel_weight is not None:
                elementwise_ce = F.binary_cross_entropy_with_logits(net_output, target_regions, reduction="none")
                combined_weight = pixel_weight if mask is None else pixel_weight * mask
                ce_loss = (elementwise_ce * combined_weight).sum() / torch.clip(combined_weight.sum(), min=1e-8)
            elif mask is not None:
                ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
            else:
                ce_loss = self.ce(net_output, target_regions)
        else:
            ce_loss = 0
        # SoftclDiceLoss has no loss_mask support (see its docstring) -- ignored
        # pixels are left as whatever target_regions already has there (0/
        # background), the same best-effort handling DC_and_CE_and_clDice_loss
        # above uses for the softmax/CE case.
        cldice_loss = self.cldice(net_output, target_regions) if self.weight_cldice != 0 else 0

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

    # DC_and_BCE_and_clDice_loss's pixel_weight path (nnUNetTrainerSmallRefinementENet's
    # gap-focused weighting) -- single-logit sigmoid output, matching SmallENet/
    # SmallRefinementENet's convention.
    bce_target = torch.randint(0, 2, (batch, 1, height, width)).float()
    bce_logits = torch.randn(batch, 1, height, width, requires_grad=True)
    bce_combo_fn = DC_and_BCE_and_clDice_loss(
        bce_kwargs={}, soft_dice_kwargs={"batch_dice": True, "do_bg": True, "smooth": 1e-5, "ddp": False},
        cldice_kwargs={"num_iter": 3, "do_bg": True},
        weight_ce=1.0, weight_dice=1.0, weight_cldice=1.0,
    )
    unweighted = bce_combo_fn(bce_logits, bce_target)
    unweighted.backward()
    print("DC_and_BCE_and_clDice_loss (no pixel_weight) value:", unweighted.item())
    print("bce_logits.grad is not None:", bce_logits.grad is not None)

    bce_logits2 = torch.randn(batch, 1, height, width, requires_grad=True)
    # Weight only a small corner strongly, like a gap region within a mostly-agreeing patch.
    pixel_weight = torch.ones(batch, 1, height, width)
    pixel_weight[:, :, :8, :8] = 9.0
    weighted = bce_combo_fn(bce_logits2, bce_target, pixel_weight=pixel_weight)
    weighted.backward()
    print("DC_and_BCE_and_clDice_loss (with pixel_weight) value:", weighted.item())
    print("bce_logits2.grad is not None:", bce_logits2.grad is not None)
    print("bce_logits2.grad has NaNs:", torch.isnan(bce_logits2.grad).any().item())
    corner_grad_mag = bce_logits2.grad[:, :, :8, :8].abs().mean().item()
    rest_grad_mag = bce_logits2.grad[:, :, 8:, 8:].abs().mean().item()
    print(f"mean |grad| inside weighted corner: {corner_grad_mag:.5f}, outside: {rest_grad_mag:.5f} "
          f"(inside should be noticeably larger)")
    assert corner_grad_mag > rest_grad_mag, "pixel_weight should make the weighted region's BCE gradient larger"
    print("pixel_weight self-test PASSED")
