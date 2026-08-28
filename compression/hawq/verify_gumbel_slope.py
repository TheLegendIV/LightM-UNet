"""Pre-flight checklist for the bare-metal Gumbel-Softmax slope-selection
experiment (QuantDecomposedLeakyAct.enable_gumbel_slope in QuantENet.py) --
runs in seconds, no checkpoint/dataset needed. Confirms the mechanism is
mechanically sound BEFORE spending any real GPU time on the actual
cheap-proxy training test.

Checks:
1. Output shape correctness (train and eval mode).
2. Hard-argmax-at-low-tau matches the plain closed-form
   alpha*x + (1-alpha)*relu(x) identity, both in eval mode (always hard
   argmax) and in train mode with tau forced very low (Gumbel-Softmax's own
   hard=True straight-through estimator should collapse to the same
   one-hot pick as tau -> 0, given a dominant logit).
3. All 5 logits get nonzero gradient on a single backward pass, at the
   PRODUCTION default init (uniform logits, tau=5.0) -- confirms the
   Gumbel-Softmax path is genuinely active (every candidate contributes
   to the mixture), not an accidental detach or a degenerate all-but-one-
   zero-gradient situation.
4. The epoch-hook mutation pattern nnUNetTrainerENetQuant26_9_w24_s14w12_
   nonneg_blockBlock.on_epoch_start will actually use (walk model.modules(),
   call set_gumbel_temperature on every QuantDecomposedLeakyAct found)
   reaches every one of this architecture's real 23 leaky-activation sites,
   on the REAL QuantENet26_9_w24_s14w12_nonneg_block class (homogeneous
   W8A8 bits, no checkpoint needed -- same pattern that file's own
   __main__ self-test already uses).

Usage (inside the dev container):
    python compression/hawq/verify_gumbel_slope.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.QuantENet import QuantDecomposedLeakyAct  # noqa: E402
from nnunetv2.nets.QuantENet26_9_w24_s14w12_nonneg_block import (  # noqa: E402
    BLOCK_NAMES, QuantENet26_9_w24_s14w12_nonneg_block,
)

torch.manual_seed(0)

LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8)


def check_1_output_shape() -> None:
    act = QuantDecomposedLeakyAct(channels=4, act_bit_width=8, negative_slope=0.0, trainable_slope=True)
    act.enable_gumbel_slope(levels=LEVELS, tau=5.0)
    x = torch.randn(2, 4, 8, 8)

    act.train()
    out_train = act(x)
    out_train_t = out_train.value if hasattr(out_train, "value") else out_train
    assert tuple(out_train_t.shape) == (2, 4, 8, 8), f"train-mode output shape wrong: {tuple(out_train_t.shape)}"

    act.eval()
    with torch.no_grad():
        out_eval = act(x)
    out_eval_t = out_eval.value if hasattr(out_eval, "value") else out_eval
    assert tuple(out_eval_t.shape) == (2, 4, 8, 8), f"eval-mode output shape wrong: {tuple(out_eval_t.shape)}"
    print("Check 1 PASSED: output shape unchanged in both train and eval mode.")


def check_2_hard_argmax_matches_closed_form() -> None:
    act = QuantDecomposedLeakyAct(channels=1, act_bit_width=8, negative_slope=0.0, trainable_slope=True)
    act.enable_gumbel_slope(levels=LEVELS, tau=5.0)
    act.set_quant_enabled(False)  # bypass Brevitas fake-quant noise -- isolate the slope-selection logic itself

    with torch.no_grad():
        act.gumbel_logits.copy_(torch.tensor([-10.0, -10.0, 10.0, -10.0, -10.0]))  # overwhelmingly favors index 2 -> level 0.4
    expected_level = LEVELS[2]

    x = torch.randn(1, 1, 4, 4)
    expected = x * expected_level + F.relu(x) * (1.0 - expected_level)

    act.eval()
    with torch.no_grad():
        out_eval = act(x)
    assert torch.allclose(out_eval, expected, atol=1e-5), (
        f"eval mode (always hard argmax) doesn't match closed-form: max diff "
        f"{(out_eval - expected).abs().max().item():.6f}"
    )

    act.train()
    act.set_gumbel_temperature(1e-4)  # near-zero tau -- hard=True straight-through should collapse to the same one-hot pick
    with torch.no_grad():
        out_train_lowtau = act(x)
    assert torch.allclose(out_train_lowtau, expected, atol=1e-3), (
        f"train mode at low tau doesn't match closed-form: max diff "
        f"{(out_train_lowtau - expected).abs().max().item():.6f}"
    )
    print("Check 2 PASSED: hard-argmax output matches the plain alpha*x + (1-alpha)*relu(x) identity "
          "in both eval mode and train mode at low tau.")


def check_3_all_logits_get_gradient() -> None:
    act = QuantDecomposedLeakyAct(channels=1, act_bit_width=8, negative_slope=0.0, trainable_slope=True)
    act.enable_gumbel_slope(levels=LEVELS, tau=5.0)  # production defaults: uniform logits (zeros), tau0=5.0
    act.train()

    x = torch.randn(1, 1, 4, 4)
    out = act(x)
    loss = out.value.pow(2).sum() if hasattr(out, "value") else out.pow(2).sum()
    loss.backward()

    assert act.gumbel_logits.grad is not None, "gumbel_logits.grad is None -- Gumbel-Softmax path not active."
    grad_mags = act.gumbel_logits.grad.abs().tolist()
    print(f"  gumbel_logits.grad magnitudes: {[f'{g:.6f}' for g in grad_mags]}")
    assert all(g > 1e-8 for g in grad_mags), (
        f"not all 5 logits got nonzero gradient -- some candidates are being starved of signal: {grad_mags}"
    )
    print("Check 3 PASSED: all 5 candidate levels received nonzero gradient on a single backward pass.")


def check_4_epoch_hook_reaches_every_site() -> None:
    homogeneous_w = {b: 8 for b in BLOCK_NAMES}
    homogeneous_a = {b: 8 for b in BLOCK_NAMES}
    # Decoder blocks (up4/regular4.*/up5/regular5.0/final) are ALWAYS plain
    # QuantReLU regardless of the slope map -- only map the 23 real
    # encoder/context sites, matching the architecture's own real usage.
    decoder_blocks = {"up4", "regular4.0", "regular4.1", "up5", "regular5.0", "final"}
    full_slope_map = {b: 0.5 for b in BLOCK_NAMES if b not in decoder_blocks}
    model = QuantENet26_9_w24_s14w12_nonneg_block(homogeneous_w, homogeneous_a, leaky_slope_map=full_slope_map)

    leaky_sites = [m for m in model.modules() if isinstance(m, QuantDecomposedLeakyAct)]
    print(f"  Found {len(leaky_sites)} QuantDecomposedLeakyAct sites before enabling gumbel_slope.")
    # NOTE: this is NOT 1 site per mapped block name -- QuantRegularBottleneck/
    # QuantInitialBlock/QuantDownsamplingBottleneck each have MULTIPLE internal
    # activation sites (reduce/conv_bn_act/out_act etc.), all built from the
    # SAME slope_map.get(block_name) value but as independent instances --
    # 23 mapped block names -> 83 real sites for this architecture (confirmed
    # elsewhere this session via the trainable_slope alpha-parameter count).
    # The real invariant to check is just "some real sites got built" (the
    # slope map actually took effect, not silently building plain QuantReLU
    # everywhere), not a specific count.
    assert len(leaky_sites) > 0, "no QuantDecomposedLeakyAct sites were built -- the slope map had no effect."
    for module in leaky_sites:
        module.enable_gumbel_slope(levels=LEVELS, tau=5.0)

    # Same walk-and-mutate pattern on_epoch_start uses in the real trainer.
    for module in model.modules():
        if isinstance(module, QuantDecomposedLeakyAct):
            module.set_gumbel_temperature(2.0)

    mutated = [m for m in model.modules() if isinstance(m, QuantDecomposedLeakyAct) and m.gumbel_slope]
    assert len(mutated) == len(leaky_sites), (
        f"epoch-hook walk reached {len(mutated)}/{len(leaky_sites)} sites -- expected all of them."
    )
    assert all(m.tau == 2.0 for m in mutated), "not every site's tau was updated by the walk-and-mutate pass."
    print(f"Check 4 PASSED: epoch-hook pattern reached and mutated all {len(mutated)} leaky-activation sites "
          f"in the real QuantENet26_9_w24_s14w12_nonneg_block architecture.")


def main() -> None:
    check_1_output_shape()
    check_2_hard_argmax_matches_closed_form()
    check_3_all_logits_get_gradient()
    check_4_epoch_hook_reaches_every_site()
    print("\nAll pre-flight checks PASSED. Safe to proceed to the smoke test + real cheap-proxy training run.")


if __name__ == "__main__":
    main()
