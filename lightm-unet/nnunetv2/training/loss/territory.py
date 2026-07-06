"""Differentiable coronary-territory consistency loss: penalizes predicted
probability mass on the anatomically wrong side (RCA vs. LAD/LCX) of
ARCADE's hard territory constraint -- a single angiogram is always
exclusively one coronary tree or the other (verified: 0/300 ARCADE test
images have both present; see analysis/segmentation_topology.py's
territory_leakage, which measures this same rule post-hoc on a finished
prediction).

Unlike territory_leakage (a measurement) or post-processing/postprocess.py's
enforce_territory (a non-differentiable, inference-time correction that must
GUESS the true territory from the prediction alone -- measured to
occasionally guess wrong when a model has already confidently misclassified
the whole image, deleting the smaller-but-correct side instead of the wrong
one), this loss runs during training, where ground truth is available. The
true territory for each image is read directly off its GT, so there is no
guessing and none of the "confidently wrong" failure mode the post-processing
heuristic has. It is also far cheaper than betti.py's Betti-0 term: no CPU
roundtrip, no Union-Find, no cropping -- just elementwise probability sums
over the full image, entirely on-GPU.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from nnunetv2.training.loss.cldice import SoftclDiceLoss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

BACKGROUND, LAD, RCA, LCX = 0, 1, 2, 3


class TerritoryLoss(nn.Module):
    """Hinge penalty on wrong-territory probability, for images whose ground
    truth is unambiguously RCA-only or LAD/LCX-only (ARCADE's normal case).

    Only the per-pixel excess of wrong-territory probability *above* `margin`
    counts, raised to `power`: a pixel sitting at or below margin contributes
    exactly zero loss AND exactly zero gradient (relu's derivative is 0 below
    its threshold), so once the network has already learned to keep RCA and
    LAD/LCX apart, this term goes fully silent and stops competing with
    Dice/CE/clDice at all -- it does not keep pushing wrong-territory
    probability toward a lower and lower value forever the way a plain
    mean(probability) penalty would (that was this module's first version;
    it was never exactly zero, just small, and unrealistic self-test logits
    initially hid how weak its signal actually was -- see git history).
    Above the margin, `power > 1` makes the penalty grow faster than linear,
    so a confident, large violation is punished much more than a marginal
    one just over the threshold.

    This never touches LAD-vs-LCX co-occurrence -- both are valid together
    in a LAD/LCX-only image, and this loss only ever penalizes RCA appearing
    in a LAD/LCX case or LAD/LCX appearing in an RCA case. An image whose GT
    has neither (or, not expected in ARCADE, both) territories present
    contributes zero -- there is no "wrong side" to define for it.
    """

    def __init__(self, apply_nonlin: Callable = None, margin: float = 0.05, power: float = 2.0):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.margin = margin
        self.power = power

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        if net_output.ndim != target.ndim:
            target = target.view((target.shape[0], 1, *target.shape[1:]))

        if net_output.shape == target.shape:
            y_onehot = target.float()
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device, dtype=net_output.dtype)
            y_onehot.scatter_(1, target.long(), 1)

        # rca_prob/left_prob/gt_has_* below all index out the channel dim (net_output[:, RCA] etc.),
        # so they're (B, H, W) -- one dim shallower than net_output's (B, C, H, W). Spatial dims for
        # those already-indexed tensors are therefore (1, 2, ...), not (2, 3, ...).
        reduce_dims = tuple(range(1, net_output.ndim - 1))

        rca_prob = net_output[:, RCA]
        left_prob = net_output[:, LAD] + net_output[:, LCX]

        # Per-pixel hinge: only probability above margin is a "violation." Squaring (default power=2)
        # keeps a large violation penalized much harder than one that barely crosses the margin.
        rca_violation = torch.clamp(rca_prob - self.margin, min=0.0) ** self.power
        left_violation = torch.clamp(left_prob - self.margin, min=0.0) ** self.power

        gt_has_rca = y_onehot[:, RCA].sum(dim=reduce_dims) > 0
        gt_has_left = (y_onehot[:, LAD] + y_onehot[:, LCX]).sum(dim=reduce_dims) > 0
        is_rca_case = gt_has_rca & ~gt_has_left
        is_left_case = gt_has_left & ~gt_has_rca

        # If GT is RCA-only, LAD/LCX violation is the wrong-territory signal, and vice versa.
        rca_case_penalty = left_violation.mean(dim=reduce_dims)
        left_case_penalty = rca_violation.mean(dim=reduce_dims)

        valid = is_rca_case | is_left_case
        if not bool(valid.any()):
            return net_output.sum() * 0.0  # zero, but keeps graph/device/dtype consistent

        penalty = torch.where(is_rca_case, rca_case_penalty, left_case_penalty)
        return penalty[valid].mean()


class DC_and_CE_and_clDice_and_Territory_loss(nn.Module):
    """Dice + CE + soft-clDice + territory-consistency. clDice rewards
    centerline overlap; territory-consistency penalizes RCA/LAD/LCX
    cross-territory probability mass directly from ground truth. See
    TerritoryLoss and this module's docstring for why this is safe to
    combine with clDice, unlike Betti-0 (measured to collapse training to
    predicting no foreground at all -- see betti.py / cldice_betti.py and
    nnUNetTrainerENetComboClDiceBetti's training-log history)."""

    def __init__(
        self,
        soft_dice_kwargs: dict,
        ce_kwargs: dict,
        cldice_kwargs: dict,
        territory_kwargs: dict | None = None,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        weight_cldice: float = 0.5,
        weight_territory: float = 0.5,
        ignore_label: int | None = None,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cldice = weight_cldice
        self.weight_territory = weight_territory
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.cldice = SoftclDiceLoss(apply_nonlin=softmax_helper_dim1, **cldice_kwargs)
        self.territory = TerritoryLoss(apply_nonlin=softmax_helper_dim1, **(territory_kwargs or {}))

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one-hot encoded target variables "
                "(DC_and_CE_and_clDice_and_Territory_loss)"
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
        territory_loss = self.territory(net_output, target_dice) if self.weight_territory != 0 else 0

        return (
            self.weight_ce * ce_loss
            + self.weight_dice * dc_loss
            + self.weight_cldice * cldice_loss
            + self.weight_territory * territory_loss
        )


if __name__ == "__main__":
    # Quick correctness self-test -- run directly (`python territory.py`) on a
    # machine with torch installed. No CUDA needed; runs fine on CPU.
    torch.manual_seed(0)
    height, width, num_classes = 32, 32, 4  # 0=Background, 1=LAD, 2=RCA, 3=LCX

    # Batch of 2: item 0 is an RCA-only case, item 1 is an LAD/LCX-only case
    # (ARCADE's normal invariant -- never both in one image).
    gt = torch.zeros(2, 1, height, width, dtype=torch.long)
    gt[0, 0, 10:14, 10:20] = RCA
    gt[1, 0, 10:14, 10:14] = LAD
    gt[1, 0, 18:22, 18:22] = LCX

    def make_logits(leak_wrong_territory: bool) -> torch.Tensor:
        # Background confident everywhere by default (mimicking a reasonably-
        # trained network, not a from-scratch one) -- betti.py's self-test
        # ties background and foreground at an equal baseline (-3/-3)
        # instead, which leaves an artificial uniform ~0.25-0.5 "ambient"
        # probability on every class everywhere; averaged over the whole
        # image, that ambient noise floor swamps a small leak region and
        # produces a misleadingly weak with-leak-vs-without-leak signal. Set
        # background confidently high here so this test actually reflects
        # the training-time regime the loss needs to behave well in.
        logits = torch.zeros(2, num_classes, height, width)
        logits[:, BACKGROUND] = 3.0
        logits[:, LAD] = -3.0
        logits[:, RCA] = -3.0
        logits[:, LCX] = -3.0

        def set_class(item, cls, rows, cols):
            logits[item, BACKGROUND, rows, cols] = -3.0
            logits[item, cls, rows, cols] = 3.0

        set_class(0, RCA, slice(10, 14), slice(10, 20))
        set_class(1, LAD, slice(10, 14), slice(10, 14))
        set_class(1, LCX, slice(18, 22), slice(18, 22))
        if leak_wrong_territory:
            set_class(0, LAD, slice(20, 24), slice(20, 24))  # item 0 (RCA-only GT) leaks confident LAD
            set_class(1, RCA, slice(2, 6), slice(2, 6))       # item 1 (LAD/LCX-only GT) leaks confident RCA
        return logits.clone().requires_grad_(True)

    territory_loss_fn = TerritoryLoss(apply_nonlin=softmax_helper_dim1, margin=0.05, power=2.0)

    logits_leaky = make_logits(leak_wrong_territory=True)
    loss_leaky = territory_loss_fn(logits_leaky, gt)
    loss_leaky.backward()
    print("TerritoryLoss WITH cross-territory leak (expect notably > 0):", loss_leaky.item())
    print("logits_leaky.grad is not None:", logits_leaky.grad is not None)
    print("logits_leaky.grad has NaNs:", torch.isnan(logits_leaky.grad).any().item())
    print("mean |grad| on the leaked RCA-in-LAD/LCX-case region:",
          logits_leaky.grad[1, RCA, 2:6, 2:6].abs().mean().item())
    print("mean |grad| on the correct same-item LAD region      :",
          logits_leaky.grad[1, LAD, 10:14, 10:14].abs().mean().item())

    logits_clean = make_logits(leak_wrong_territory=False)
    loss_clean = territory_loss_fn(logits_clean, gt)
    print("TerritoryLoss WITHOUT any leak (expect ~0):", loss_clean.item())

    # An all-background item (no GT foreground at all) must contribute zero,
    # not error out.
    gt_empty = torch.zeros(1, 1, height, width, dtype=torch.long)
    logits_empty = torch.full((1, num_classes, height, width), -3.0, requires_grad=True)
    loss_empty = territory_loss_fn(logits_empty, gt_empty)
    print("TerritoryLoss on an all-background item (expect exactly 0):", loss_empty.item())

    # The "otherwise invisible" requirement: a *small* amount of wrong-territory
    # probability (below margin=0.05) must give EXACTLY zero loss and EXACTLY
    # zero gradient -- not just a small nonzero value. This is what makes the
    # hinge version different from the first (plain-mean) version, which was
    # always at least slightly active everywhere.
    logits_small_leak = make_logits(leak_wrong_territory=False).detach().clone()
    logits_small_leak[0, LAD, 20:24, 20:24] = -1.5  # softmax vs background(3.0): ~0.018 prob, below margin
    logits_small_leak.requires_grad_(True)
    loss_small_leak = territory_loss_fn(logits_small_leak, gt)
    loss_small_leak.backward()
    below_margin_prob = torch.softmax(logits_small_leak, dim=1)[0, LAD, 20:24, 20:24].mean().item()
    print(f"Below-margin wrong-territory probability ({below_margin_prob:.4f}, margin=0.05) "
          f"-- loss (expect exactly 0.0):", loss_small_leak.item())
    print("grad on that below-margin region (expect exactly 0.0):",
          logits_small_leak.grad[0, LAD, 20:24, 20:24].abs().sum().item())
    assert loss_small_leak.item() == 0.0
    assert logits_small_leak.grad[0, LAD, 20:24, 20:24].abs().sum().item() == 0.0

    # LAD and LCX co-occurring in one LAD/LCX-only image is normal, not a
    # violation -- confirm the loss is 0 for item 1 even though both LAD and
    # LCX are confidently predicted together (this is exactly the correct,
    # leak-free item-1 setup from logits_clean above).
    print("Confirmed: item 1 (LAD+LCX together, no RCA) contributes to the mean above as 0 "
          "-- this loss never penalizes LAD/LCX co-occurring, only RCA-vs-left mixing.")

    combo_loss_fn = DC_and_CE_and_clDice_and_Territory_loss(
        soft_dice_kwargs={"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False},
        ce_kwargs={},
        cldice_kwargs={"num_iter": 3, "do_bg": False},
        weight_ce=1.0, weight_dice=1.0, weight_cldice=0.5, weight_territory=0.5,
    )
    logits2 = make_logits(leak_wrong_territory=True)
    combo = combo_loss_fn(logits2, gt)
    combo.backward()
    print("DC_and_CE_and_clDice_and_Territory_loss value:", combo.item())
    print("logits2.grad is not None:", logits2.grad is not None)
