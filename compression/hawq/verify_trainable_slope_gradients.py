"""Step 0 of the stuck-QAT debugging plan (compression/configs or session
notes -- see the plan file for full context): verifies trainable_slope's
gradient is not just NONZERO (already confirmed elsewhere this session) but
NUMERICALLY CORRECT, by comparing against a de-sugared reference computation
that bypasses Brevitas's own QuantTensor.__mul__/__add__ operator overloads.

WHY: alpha is a plain nn.Parameter multiplying a Brevitas QuantTensor via
ordinary tensor arithmetic (see QuantDecomposedLeakyAct.forward in
QuantENet.py). Every prior real training config used this operator with a
FROZEN buffer as the multiplicand -- this session is the first time it's
ever been multiplied by something with requires_grad=True. "Gradient
exists and changes after an optimizer step" doesn't rule out a subtly
WRONG gradient (e.g. a scale-tracking side effect specific to QuantTensor's
operator overloading that only manifests for a trainable operand). This
script isolates exactly that question, with no checkpoint/dataset needed --
runs in seconds.

METHOD: build the REAL class under test with channels=1 (single scalar
slope, hand-checkable). Run the real forward() (path A, alpha is a live
nn.Parameter, gradient flows through Brevitas's QuantTensor arithmetic).
Separately, recompute the exact same pre_quant/act_pos outputs via a FRESH
forward call (deterministic in eval mode, same values), detach them
(alpha's gradient only depends on their VALUES, not their own upstream
graph -- x_q/pos don't depend on alpha), and redo the alpha-weighted-sum
step with a FRESH parameter (path B, plain torch.Tensor arithmetic, no
QuantTensor involved), then re-quantize through the SAME out_quant module
so every other step matches. Compare path A's alpha.grad against path B's
alpha2.grad.

PASS: they match (within float tolerance) -- Brevitas's QuantTensor
arithmetic is transparent for a trainable multiplicand, the gradient is
trustworthy, "wrong gradient" is ruled out.
FAIL: they diverge -- a genuine, previously-hidden bug in how Brevitas's
operator overloading handles a trainable (vs. frozen-buffer) multiplicand.
Would immediately become the top-priority fix, ahead of every other
debugging step, and would mean every trainable_slope=True run this session
is unreliable by construction.

Usage (inside the dev container):
    python compression/hawq/verify_trainable_slope_gradients.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.QuantENet import QuantDecomposedLeakyAct  # noqa: E402

torch.manual_seed(0)


def main() -> None:
    negative_slope = 0.25
    act = QuantDecomposedLeakyAct(channels=1, act_bit_width=8, negative_slope=negative_slope, trainable_slope=True)
    act.eval()  # deterministic quantization -- no observer/calibration-mode side effects between the two forward calls below

    x = torch.randn(1, 1, 4, 4)

    # --- Path A: the REAL forward path, exactly as production code calls it ---
    out_a = act(x)
    loss_a = out_a.value.pow(2).sum()
    loss_a.backward()
    grad_a = act.alpha.grad.clone()

    # --- Path B: de-sugared reference -- same quantized x_q/pos VALUES (a
    # fresh, separate forward call; deterministic in eval mode so numerically
    # identical to path A's own x_q/pos), but the alpha-weighted-sum step
    # uses PLAIN torch.Tensor arithmetic instead of QuantTensor.__mul__/__add__,
    # and a FRESH parameter (alpha's gradient doesn't depend on x_q/pos's own
    # upstream graph -- they don't depend on alpha at all -- so detaching them
    # to plain float constants changes nothing about d(loss)/d(alpha)). ---
    with torch.no_grad():
        x_q = act.pre_quant(x)
        pos = act.act_pos(x_q)
    x_q_val = x_q.value.detach()
    pos_val = pos.value.detach()

    alpha2 = nn.Parameter(torch.full_like(act.alpha.detach(), negative_slope))
    combined = x_q_val * alpha2 + pos_val * (1.0 - alpha2)
    out_b = act.out_quant(combined)
    loss_b = out_b.value.pow(2).sum()
    loss_b.backward()
    grad_b = alpha2.grad.clone()

    # --- Bonus cross-check: naive closed-form gradient IGNORING out_quant's
    # own quantization (loss = sum((x_q*a + pos*(1-a))^2) w.r.t. a, evaluated
    # at a=negative_slope, treating x_q/pos as constants) -- a third,
    # independent sanity check, not expected to match exactly (out_quant's
    # own re-quantization contributes its own gradient term) but should be
    # in the same ballpark/sign. ---
    a0 = negative_slope
    pre_quant_out = x_q_val * a0 + pos_val * (1.0 - a0)
    naive_grad = (2 * pre_quant_out * (x_q_val - pos_val)).sum()

    print(f"Path A (real forward, QuantTensor arithmetic):      alpha.grad = {grad_a.flatten().tolist()}")
    print(f"Path B (de-sugared, plain tensor arithmetic):       alpha.grad = {grad_b.flatten().tolist()}")
    print(f"Naive closed-form (ignoring out_quant's own quant): d(loss)/d(alpha) ~= {naive_grad.item():.6f}")

    max_abs_diff = (grad_a - grad_b).abs().max().item()
    print(f"\nMax abs difference between Path A and Path B: {max_abs_diff:.8f}")

    tolerance = 1e-5
    if max_abs_diff < tolerance:
        print(f"PASS: gradients match within tolerance ({tolerance}) -- Brevitas's QuantTensor "
              f"arithmetic is transparent for a trainable multiplicand. trainable_slope's gradient is trustworthy.")
    else:
        print(f"FAIL: gradients diverge by {max_abs_diff:.8f}, exceeding tolerance ({tolerance}). "
              f"This is a genuine discrepancy between Brevitas's QuantTensor operator overloading and "
              f"plain-tensor arithmetic for a trainable multiplicand -- investigate before trusting any "
              f"trainable_slope=True training run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
