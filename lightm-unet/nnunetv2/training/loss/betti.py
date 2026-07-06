"""Betti-0 (connected-component) topological loss, penalizing disconnected /
fragmented predictions directly via 0-dimensional persistent homology --
complementary to clDice (cldice.py), which rewards centerline overlap but
never explicitly counts or penalizes the *number* of disconnected pieces (a
prediction can have great clDice and still be fragmented into several
same-class islands that overlap the true centerline well individually).

Follows Hu et al., "Topology-Preserving Deep Image Segmentation" (NeurIPS
2019), restricted to dimension 0 (connected components / Betti-0), since the
goal here is specifically "penalize disconnected regions." Dimension 1
(loops/holes) would need a full cubical-complex persistent homology
implementation (e.g. via gudhi/ripser) and is not implemented here.

How it works, per foreground class channel of one image:
  1. Treat the predicted probability map as a superlevel-set filtration: as a
     threshold t sweeps from 1 down to 0, pixels with prob >= t "appear" in
     that order, and connected components of already-appeared pixels are
     tracked with a Union-Find structure. This *is* 0-dimensional persistent
     homology on a cubical complex -- no approximation.
  2. Each component is "born" at the probability value of the pixel that
     started it, and "dies" (merges into an older component, elder rule) at
     the probability value of whichever pixel first connects it to that
     older component. Persistence = birth - death; the one component that
     never merges away (eventually the whole active region) has
     maximal/unbounded persistence.
  3. The ground truth mask for this class has some true connected-component
     count k (usually 1 for a single vessel tree, occasionally more if
     genuinely disjoint in-frame -- computed fresh per image via
     scipy.ndimage.label, not assumed, and always from the FULL image, not
     the patch_size crop below -- see that section's comment for why this
     matters). The top k predicted components by persistence are treated as
     the real anatomical pieces; anything beyond that is a spurious fragment.
  4. The loss pushes the probability at each spurious component's birth
     pixel toward 0 -- directly discouraging the network from keeping that
     island alive as a separate piece.

This only penalizes over-fragmentation (too many pieces). It does not
implement the symmetric "encourage a death pixel's probability up" term for
under-segmenting a genuinely-disconnected GT structure, since that's not
what "penalize disconnected regions" is asking for.

Cost warning: Union-Find over a superlevel-set filtration is an inherently
sequential, per-pixel algorithm and cannot be vectorized/batched on GPU the
way Dice/CE/clDice can. It runs on CPU, per class-channel, per batch item.
`prob_floor` skips negligible-probability pixels and `patch_size` caps the
region considered per call (a random crop, not the full image) to bound
worst-case cost regardless of image resolution -- lower patch_size trades
per-step topological signal quality for speed. Expect this to be the most
expensive term in any loss that includes it.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from scipy import ndimage as ndi
from torch import nn

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

_NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_NEIGHBORS_4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.birth_pixel = [-1] * n
        self.birth_value = [0.0] * n

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union_absorb(self, survivor_root: int, dying_root: int) -> None:
        """Merge dying_root into survivor_root. survivor_root keeps its birth info."""
        self.parent[dying_root] = survivor_root


def _persistence_diagram(prob_map: np.ndarray, connectivity: int = 8, prob_floor: float = 0.01) -> list[tuple]:
    """0-dim persistence diagram of prob_map's superlevel-set filtration.

    Returns a list of (birth_pixel_flat_idx, birth_value, death_value) for
    every component, in the order encountered. death_value is None for
    component(s) that survive to the end (never merge away).
    """
    height, width = prob_map.shape
    flat = prob_map.reshape(-1)
    n = flat.size

    candidate_idx = np.flatnonzero(flat >= prob_floor)
    if candidate_idx.size == 0:
        return []
    order = candidate_idx[np.argsort(-flat[candidate_idx], kind="stable")]

    active = np.zeros(n, dtype=bool)
    uf = _UnionFind(n)
    offsets = _NEIGHBORS_8 if connectivity == 8 else _NEIGHBORS_4
    diagram: list[tuple] = []

    for idx in order:
        idx = int(idx)
        y, x = divmod(idx, width)
        active[idx] = True

        neighbor_roots = set()
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                nidx = ny * width + nx
                if active[nidx]:
                    neighbor_roots.add(uf.find(nidx))

        if not neighbor_roots:
            uf.birth_pixel[idx] = idx
            uf.birth_value[idx] = float(flat[idx])
            continue

        roots = list(neighbor_roots)
        survivor = max(roots, key=lambda r: uf.birth_value[r])  # elder rule: oldest (highest birth) survives
        for r in roots:
            if r != survivor:
                diagram.append((uf.birth_pixel[r], uf.birth_value[r], float(flat[idx])))
                uf.union_absorb(survivor, r)
        uf.parent[idx] = survivor

    roots_seen = {uf.find(int(idx)) for idx in order}
    for r in roots_seen:
        diagram.append((uf.birth_pixel[r], uf.birth_value[r], None))

    return diagram


def _spurious_birth_pixels(
    prob_map: np.ndarray, expected_components: int, connectivity: int = 8, prob_floor: float = 0.01,
) -> list[int]:
    """Flat pixel indices of components ranked beyond expected_components by
    persistence -- the excess, spurious fragments to penalize."""
    diagram = _persistence_diagram(prob_map, connectivity=connectivity, prob_floor=prob_floor)
    if len(diagram) <= expected_components:
        return []

    def persistence(bar: tuple) -> float:
        birth_idx, birth_value, death_value = bar
        return birth_value if death_value is None else (birth_value - death_value)

    ranked = sorted(diagram, key=persistence, reverse=True)
    return [bar[0] for bar in ranked[expected_components:]]


def expected_components_per_channel(gt_onehot_np: np.ndarray) -> np.ndarray:
    """Ground-truth connected-component count per (batch, class), from the
    FULL image -- gt_onehot_np: (B, C, H, W) array, values > 0.5 = foreground.

    Deliberately takes the full image rather than any sub-crop: see the
    "expected_components must come from the FULL image" comment in
    Betti0Loss.forward for why using a small crop's own GT here silently
    breaks training (measured: collapses predicted foreground to nothing).
    """
    batch, n_classes = gt_onehot_np.shape[:2]
    expected = np.zeros((batch, n_classes), dtype=int)
    for b in range(batch):
        for c in range(n_classes):
            gt_mask = gt_onehot_np[b, c] > 0.5
            expected[b, c] = 0 if not gt_mask.any() else ndi.label(gt_mask)[1]
    return expected


class Betti0Loss(nn.Module):
    """Penalizes excess (beyond ground truth's connected-component count)
    0-dimensional topological features in a predicted probability map --
    i.e. directly discourages fragmentation, unlike Dice or clDice, neither
    of which explicitly counts pieces.

    Computed per foreground class (background excluded, matching this
    repo's do_bg=False convention) and per batch item: finds the ground
    truth's connected-component count, finds the predicted probability
    map's persistence diagram, and penalizes probability at the birth pixel
    of every predicted component beyond that count.
    """

    def __init__(
        self,
        apply_nonlin: Callable = None,
        do_bg: bool = False,
        connectivity: int = 8,
        prob_floor: float = 0.01,
        patch_size: int | None = 64,
        max_components_penalized: int = 64,
    ):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.do_bg = do_bg
        self.connectivity = connectivity
        self.prob_floor = prob_floor
        self.patch_size = patch_size
        self.max_components_penalized = max_components_penalized

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

        if not self.do_bg:
            net_output = net_output[:, 1:]
            y_onehot = y_onehot[:, 1:]

        if net_output.ndim != 4:
            raise NotImplementedError("Betti0Loss currently only supports 2D (B, C, H, W) inputs.")

        batch, n_classes, height, width = net_output.shape

        # expected_components must come from the FULL image's GT, not the crop
        # taken below. ARCADE vessels are thin and sparse, so a small random
        # crop very often contains none of a given class's pixels even though
        # that class is genuinely present elsewhere in the same image. Deriving
        # expected_components from the crop's own (locally empty) GT would then
        # judge that class "absent here" and penalize any emerging correct
        # probability inside the crop as spurious -- since most crops of a
        # sparse class are "empty" purely by chance, this fires far more often
        # than it should and was measured to collapse training to predicting no
        # foreground at all (see nnUNetTrainerENetComboClDiceBetti's smoke-test
        # history: clDice alone reached ~0.74 mean Dice; adding Betti-0 with the
        # crop-derived expected_components drove Dice to exactly 0.0 in every
        # case, at both weight=0.5 and weight=1.0). Using the full image here
        # avoids that false "absent" verdict; the crop below still bounds the
        # cost of the expensive part (the per-pixel persistence diagram).
        with torch.no_grad():
            gt_full_np = y_onehot.detach().cpu().numpy()
        expected_components = expected_components_per_channel(gt_full_np)

        # Random crop bounds the Union-Find cost regardless of image size --
        # see module docstring's cost warning. Only the probability map is
        # cropped; expected_components above already reflects the full image.
        if self.patch_size is not None and (height > self.patch_size or width > self.patch_size):
            ph = min(self.patch_size, height)
            pw = min(self.patch_size, width)
            y0 = int(torch.randint(0, height - ph + 1, (1,)).item())
            x0 = int(torch.randint(0, width - pw + 1, (1,)).item())
            net_output_patch = net_output[:, :, y0:y0 + ph, x0:x0 + pw]
        else:
            net_output_patch = net_output

        with torch.no_grad():
            prob_np = net_output_patch.detach().cpu().numpy()

        penalty_terms = []
        for b in range(batch):
            for c in range(n_classes):
                spurious_idx = _spurious_birth_pixels(
                    prob_np[b, c], int(expected_components[b, c]),
                    connectivity=self.connectivity, prob_floor=self.prob_floor,
                )
                if not spurious_idx:
                    continue
                spurious_idx = spurious_idx[: self.max_components_penalized]

                pw_ = prob_np[b, c].shape[1]
                ys = [i // pw_ for i in spurious_idx]
                xs = [i % pw_ for i in spurious_idx]
                penalty_terms.append(net_output_patch[b, c, ys, xs].pow(2).sum())

        if not penalty_terms:
            return net_output.sum() * 0.0  # zero, but keeps graph/device/dtype consistent

        return torch.stack(penalty_terms).sum() / (batch * n_classes)


class DC_and_CE_and_Betti_loss(nn.Module):
    """Dice + CE + Betti-0. The Dice+CE part is identical to
    compound_losses.DC_and_CE_loss; Betti-0 is added as a third weighted
    term to additionally penalize the number of disconnected fragments,
    which Dice+CE alone don't score."""

    def __init__(
        self,
        soft_dice_kwargs: dict,
        ce_kwargs: dict,
        betti_kwargs: dict,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        weight_betti: float = 1.0,
        ignore_label: int | None = None,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_betti = weight_betti
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.betti = Betti0Loss(apply_nonlin=softmax_helper_dim1, **betti_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one-hot encoded target variables (DC_and_CE_and_Betti_loss)"
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
        betti_loss = self.betti(net_output, target_dice) if self.weight_betti != 0 else 0

        return self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_betti * betti_loss


if __name__ == "__main__":
    # Quick correctness self-test -- run directly (`python betti.py`) on a
    # machine with torch installed. No CUDA needed; runs fine on CPU.
    # Builds a GT with exactly 1 component, and a prediction with that same
    # component PLUS a separate spurious blob -- checks that Betti0Loss finds
    # a nonzero penalty, that it drops to (near) zero once the spurious blob
    # is removed, and that gradients flow.
    torch.manual_seed(0)
    height, width, num_classes = 40, 40, 2  # class 0 = background, class 1 = foreground

    gt = torch.zeros(1, 1, height, width, dtype=torch.long)
    gt[0, 0, 10:14, 10:30] = 1  # one horizontal bar: 1 connected component

    def make_logits(with_spurious_blob: bool) -> torch.Tensor:
        logits = torch.full((1, num_classes, height, width), -3.0)
        logits[0, 1, 10:14, 10:30] = 3.0  # confident, correct main bar
        if with_spurious_blob:
            logits[0, 1, 25:28, 5:8] = 3.0  # confident, disconnected extra blob
        return logits.clone().requires_grad_(True)

    betti_loss_fn = Betti0Loss(apply_nonlin=softmax_helper_dim1, patch_size=None, prob_floor=0.01)

    logits_with_blob = make_logits(with_spurious_blob=True)
    loss_with_blob = betti_loss_fn(logits_with_blob, gt)
    loss_with_blob.backward()
    print("Loss WITH spurious blob:", loss_with_blob.item())
    print("logits_with_blob.grad is not None:", logits_with_blob.grad is not None)
    print("logits_with_blob.grad has NaNs:", torch.isnan(logits_with_blob.grad).any().item())
    # Gradient should be concentrated on the spurious blob region (pushing its logits down),
    # and near-zero on the correct main bar.
    print("mean |grad| on spurious blob region:", logits_with_blob.grad[0, 1, 25:28, 5:8].abs().mean().item())
    print("mean |grad| on correct main bar    :", logits_with_blob.grad[0, 1, 10:14, 10:30].abs().mean().item())

    logits_clean = make_logits(with_spurious_blob=False)
    loss_clean = betti_loss_fn(logits_clean, gt)
    print("Loss WITHOUT spurious blob (expect ~0):", loss_clean.item())

    # Regression test for the full-image vs. crop expected_components bug
    # (see Betti0Loss.forward's comment): a crop that happens to miss a
    # class's only GT component must NOT cause that class to be treated as
    # "absent" -- expected_components_per_channel must always report it
    # using the full image, independent of patch_size cropping.
    gt_np = gt.numpy().astype(np.float32)
    gt_onehot_np = np.stack([1 - gt_np[:, 0], gt_np[:, 0]], axis=1)  # (1, 2, H, W): bg, fg
    full_image_expected = expected_components_per_channel(gt_onehot_np)
    crop_missing_vessel = gt_onehot_np[:, :, 25:, 25:]  # bottom-right corner, vessel is at rows 10-14
    crop_expected = expected_components_per_channel(crop_missing_vessel)
    print("expected_components, full image  (class 1 should be 1):", full_image_expected[0, 1])
    print("expected_components, empty crop  (class 1 should be 0 -- this is exactly the bug: a crop"
          " missing the vessel must never be the number Betti0Loss.forward actually uses):",
          crop_expected[0, 1])
    assert full_image_expected[0, 1] == 1
    assert crop_expected[0, 1] == 0

    combo_loss_fn = DC_and_CE_and_Betti_loss(
        soft_dice_kwargs={"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False},
        ce_kwargs={},
        betti_kwargs={"patch_size": None},
        weight_ce=1.0, weight_dice=1.0, weight_betti=1.0,
    )
    logits2 = make_logits(with_spurious_blob=True)
    combo = combo_loss_fn(logits2, gt)
    combo.backward()
    print("DC_and_CE_and_Betti_loss value:", combo.item())
    print("logits2.grad is not None:", logits2.grad is not None)
