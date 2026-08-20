"""Per-stage HAWQ-style sensitivity estimator for
nnUNetTrainerENet_23_1_s19_warmstart_4c: Hutchinson-trace Hessian estimation
(weights AND activations) combined with per-candidate-bit-width
quantization error, producing the sensitivity_23_1.json that
compression/hawq/ilp_search.py consumes.

Runs entirely on the plain FP32 ENet (23_1 config, prelu_variant=
"nonneg_block", loaded from its own checkpoint) -- NOT on QuantENet23_1 --
to avoid double-backward through Brevitas's straight-through-estimator
fake-quant ops, and because sensitivity is a property of the loss landscape
near the FP32 optimum, independent of which quantization scheme is later
applied (same practice HAWQ's own papers use).

Method (Hutchinson's trace estimator via Pearlmutter's double-backward
Hessian-vector product, same technique PyHessian implements -- not vendored
here, see this repo's hawq/ directory's own README for why: HAWQ-V3 ships
no trace-computation code at all, only a notebook with hardcoded example
numbers):
    Tr(H) ~= (1/n) * sum_i  z_i^T H z_i,   z_i ~ Rademacher(+-1)
    H z_i computed via:  g = grad(loss, target, create_graph=True)
                          Hz = grad(sum(g * z_i), target)
    (never forms H explicitly -- each z_i costs one extra backward pass)

Two kinds of targets, both driven by the SAME real forward+backward pass on
real ARCADE data + nnU-Net's own DC_and_CE_loss:
  - WEIGHT targets: every nn.Conv2d/nn.ConvTranspose2d module's .weight
    tensor, grouped into the 5 HAWQ stages (see STAGE_MODULE_ATTRS below)
    and combined via a numel-weighted average into one normalized
    (trace / #params) number per stage -- same normalization convention
    hawq/ILP.ipynb's own Hutchinson_trace values use.
  - ACTIVATION targets: the single boundary tensor leaving each of the 5
    stages (captured via a forward hook, no retain_grad() needed --
    torch.autograd.grad accepts any graph tensor, not just leaves),
    normalized by that tensor's own element count.

Per-candidate-bit quantization error: a standalone symmetric-per-tensor
fake-quantize (matches Brevitas's Int8WeightPerTensorFloat convention for
weights). For activations this uses the SAME signed-symmetric convention
rather than exactly replicating Brevitas's per-site unsigned/signed choice
(QuantReLU vs QuantDecomposedLeakyAct differ there) -- a deliberate
simplification: what the ILP needs is a consistent quantizer across bit
candidates for a given tensor (so `sensitivity[b] = trace * delta[b]`
compares meaningfully across b in {2,4,8} for THAT tensor), not bit-exact
parity with the eventual deployed quantizer.

Usage:
    python compression/hawq/sensitivity.py \\
        --net-name nnUNetTrainerENet_23_1_s19_warmstart_4c \\
        --checkpoint-name checkpoint_best.pth \\
        --n-batches 8 --n-probes 10 \\
        --out-file compression/hawq/sensitivity_23_1.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss  # noqa: E402
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_23_1 import (  # noqa: E402
    BOTTLENECKS_PER_STAGE,
    CANDIDATE_BITS,
    CHANNELS,
    CONTEXT_PATTERN,
    DECODER_TYPE,
    IN_CHANNELS,
    OUT_CHANNELS,
    PRELU_VARIANT,
    SEPARABLE_DILATED,
    STAGE_BOUNDARY_ATTR,
    STAGE_MODULE_ATTRS,
    STAGE_NAMES,
    USE_ASYMMETRIC,
)

NNUNET_PREPROCESSED = REPO_ROOT / "data" / "nnUNet_preprocessed"
NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"


def build_fp32_model(checkpoint_path: Path) -> ENet:
    model = ENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        decoder_type=DECODER_TYPE, use_asymmetric=USE_ASYMMETRIC, context_pattern=CONTEXT_PATTERN,
        separable_dilated=SEPARABLE_DILATED, use_prelu=True, prelu_variant=PRELU_VARIANT,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["network_weights"], strict=True)
    return model


def load_real_batches(dataset_name: str, n_batches: int, seed: int = 0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Real preprocessed (image, seg) pairs, batch size 1 -- same rationale
    as compression/post-quantization/ptq.py's load_calibration_batches
    (varying H/W across real cases, ENet's forward pass is fully
    convolutional so tolerates it directly). seg's raw -1 ("no label",
    pre-augmentation encoding) is replaced with 0 here, replicating the
    RemoveLabelTransform(replace_with=0, remove_label=-1) step the real
    training dataloader applies (see nnUNetTrainerENet's own transform
    pipeline) -- this script bypasses the augmentation pipeline entirely
    and reads the raw preprocessed .npy directly, so that one step needs
    reproducing by hand; every other augmentation op is skipped on purpose
    (sensitivity should reflect the real data distribution, not synthetic
    augmented crops)."""
    preprocessed_dir = NNUNET_PREPROCESSED / dataset_name / "nnUNetPlans_2d"
    image_files = sorted(p for p in preprocessed_dir.glob("*.npy") if not p.name.endswith("_seg.npy"))
    if not image_files:
        raise FileNotFoundError(f"No preprocessed .npy images found under {preprocessed_dir}")
    rng = random.Random(seed)
    sampled = rng.sample(image_files, k=min(n_batches, len(image_files)))
    batches = []
    for img_path in sampled:
        seg_path = img_path.with_name(img_path.stem + "_seg.npy")
        img = torch.from_numpy(np.load(img_path)).float()
        seg = torch.from_numpy(np.load(seg_path)).long()
        seg[seg == -1] = 0
        batches.append((img, seg))
    return batches


def rademacher_like(t: torch.Tensor) -> torch.Tensor:
    return (torch.randint(0, 2, t.shape, device=t.device) * 2 - 1).to(t.dtype)


def hutchinson_traces(
    loss: torch.Tensor, targets: dict[str, torch.Tensor], n_probes: int,
) -> dict[str, float]:
    """One first-order-grad graph (create_graph=True) shared across all
    n_probes -- each probe costs exactly one extra backward pass total, not
    one per target (the standard batched-HVP trick). Returns the RAW
    (unnormalized) sum of z^T H z across probes for each target -- caller
    divides by n_probes and by the target's own element count.

    retain_graph=True on EVERY autograd.grad call here, including the last
    probe: this function is called repeatedly per batch (once per stage's
    weight targets, once per stage's activation target) against the SAME
    shared `loss` forward graph -- freeing it after this call's last probe
    would break every subsequent stage's call in that batch ("Trying to
    backward through the graph a second time"). The whole shared graph is
    only actually released once `loss`/`out` are reassigned by the next
    batch iteration (ordinary Python/autograd garbage collection), not by
    an explicit retain_graph=False anywhere in here."""
    names = list(targets.keys())
    tensors = [targets[n] for n in names]
    grads = torch.autograd.grad(loss, tensors, create_graph=True, retain_graph=True)

    totals = {n: 0.0 for n in names}
    for _probe_i in range(n_probes):
        zs = [rademacher_like(t) for t in tensors]
        gz = sum((g * z).sum() for g, z in zip(grads, zs))
        hzs = torch.autograd.grad(gz, tensors, retain_graph=True)
        for n, z, hz in zip(names, zs, hzs):
            totals[n] += (z * hz).sum().item()
    return {n: totals[n] / n_probes for n in names}


def stage_weight_targets(model: ENet) -> dict[str, dict[str, torch.Tensor]]:
    """{stage: {dotted_module_name: weight_tensor}} for every Conv2d/
    ConvTranspose2d inside that stage's top-level attributes -- filtering by
    isinstance rather than parameter name avoids picking up BatchNorm's or
    NonNegativePReLU's own `.weight` parameters, which are not convolution
    weights."""
    result: dict[str, dict[str, torch.Tensor]] = {s: {} for s in STAGE_NAMES}
    attr_to_stage = {attr: stage for stage, attrs in STAGE_MODULE_ATTRS.items() for attr in attrs}
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            continue
        top_attr = name.split(".", 1)[0]
        stage = attr_to_stage.get(top_attr)
        if stage is not None:
            result[stage][name] = module.weight
    return result


def fake_quantize_symmetric(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Signed, symmetric, per-tensor fake-quantize (matches Brevitas's
    Int8WeightPerTensorFloat convention: scale = max(|x|) / (2^(bits-1)-1),
    round-to-nearest, clamp). Used identically for both weight and
    activation deltas -- see module docstring for why activations don't
    need Brevitas's exact per-site unsigned/signed choice replicated here."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.detach().abs().max().clamp(min=1e-8) / qmax
    return (x.detach() / scale).round().clamp(-qmax, qmax) * scale


def quantization_deltas(x: torch.Tensor) -> dict[int, float]:
    return {b: (x.detach() - fake_quantize_symmetric(x, b)).pow(2).sum().item() for b in CANDIDATE_BITS}


def run_sensitivity(
    checkpoint_path: Path, dataset_name: str, n_batches: int, n_probes: int, device: str, seed: int,
) -> dict:
    model = build_fp32_model(checkpoint_path).to(device)
    model.eval()  # BN uses running stats; dropout off -- sensitivity should reflect inference-time behavior
    loss_fn = DC_and_CE_loss(
        {"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False}, {},
        weight_ce=1, weight_dice=1, ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss,
    )
    weight_targets_by_stage = stage_weight_targets(model)

    boundary_modules = {stage: dict(model.named_modules())[attr] for stage, attr in STAGE_BOUNDARY_ATTR.items()}
    captured: dict[str, torch.Tensor] = {}

    def make_hook(stage_name: str):
        def hook(_module, _inputs, output):
            captured[stage_name] = output
        return hook

    handles = [module.register_forward_hook(make_hook(stage)) for stage, module in boundary_modules.items()]

    batches = load_real_batches(dataset_name, n_batches, seed=seed)
    print(f"Loaded {len(batches)} real (image, seg) batches from {dataset_name}.")

    weight_trace_sum = {s: {n: 0.0 for n in weight_targets_by_stage[s]} for s in STAGE_NAMES}
    weight_numel = {s: {n: t.numel() for n, t in weight_targets_by_stage[s].items()} for s in STAGE_NAMES}
    act_trace_sum = {s: 0.0 for s in STAGE_NAMES}
    act_numel_sum = {s: 0 for s in STAGE_NAMES}
    weight_delta_sum = {s: {b: 0.0 for b in CANDIDATE_BITS} for s in STAGE_NAMES}
    weight_numel_total = {s: 0 for s in STAGE_NAMES}
    act_delta_sum = {s: {b: 0.0 for b in CANDIDATE_BITS} for s in STAGE_NAMES}
    act_numel_total = {s: 0 for s in STAGE_NAMES}

    n_used = 0
    for batch_i, (img, seg) in enumerate(batches):
        img, seg = img.to(device), seg.to(device)
        captured.clear()
        try:
            out = model(img)
        except RuntimeError as error:
            # Some real preprocessed images have H/W not evenly divisible by
            # ENet's stride-8 downsample factor, tripping a shape mismatch
            # in the upsample_conv decoder's residual add -- same real
            # architectural sensitivity compression/post-quantization/
            # ptq.py's calibrate() already documents and skips, independent
            # of quantization. A representative SAMPLE is enough here too.
            print(f"  [skip] batch {batch_i + 1}/{len(batches)} (shape {tuple(img.shape)}) failed forward pass: {error}")
            continue
        loss = loss_fn(out, seg)
        n_used += 1

        for stage in STAGE_NAMES:
            wt = weight_targets_by_stage[stage]
            if wt:
                traces = hutchinson_traces(loss, wt, n_probes)
                for n, tr in traces.items():
                    weight_trace_sum[stage][n] += tr
            act_traces = hutchinson_traces(loss, {stage: captured[stage]}, n_probes)
            act_trace_sum[stage] += act_traces[stage]
            act_numel_sum[stage] += captured[stage].numel()

        with torch.no_grad():
            for stage, wt in weight_targets_by_stage.items():
                for tensor in wt.values():
                    deltas = quantization_deltas(tensor)
                    for b, d in deltas.items():
                        weight_delta_sum[stage][b] += d
                    weight_numel_total[stage] += tensor.numel()
            for stage in STAGE_NAMES:
                deltas = quantization_deltas(captured[stage])
                for b, d in deltas.items():
                    act_delta_sum[stage][b] += d
                act_numel_total[stage] += captured[stage].numel()

        print(f"batch {batch_i + 1}/{len(batches)} done.")

    for h in handles:
        h.remove()

    if n_used == 0:
        raise RuntimeError("Every batch failed the forward pass -- can't estimate sensitivity at all.")
    print(f"Used {n_used}/{len(batches)} batches (some may have been skipped for shape reasons, see above).")

    report = {}
    for stage in STAGE_NAMES:
        total_w_trace = sum(weight_trace_sum[stage].values()) / n_used
        total_w_numel = sum(weight_numel[stage].values())
        trace_w = total_w_trace / max(total_w_numel, 1)
        trace_a = (act_trace_sum[stage] / n_used) / max(act_numel_sum[stage] / n_used, 1)

        delta_w = {b: weight_delta_sum[stage][b] / n_used for b in CANDIDATE_BITS}
        delta_a = {b: act_delta_sum[stage][b] / n_used for b in CANDIDATE_BITS}

        report[stage] = {
            "trace_w": trace_w,
            "trace_a": trace_a,
            "delta_w": {str(b): delta_w[b] for b in CANDIDATE_BITS},
            "delta_a": {str(b): delta_a[b] for b in CANDIDATE_BITS},
            "sensitivity_w": {str(b): trace_w * delta_w[b] for b in CANDIDATE_BITS},
            "sensitivity_a": {str(b): trace_a * delta_a[b] for b in CANDIDATE_BITS},
            "n_weight_tensors": len(weight_targets_by_stage[stage]),
            "n_weight_params": total_w_numel,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--net-name", default="nnUNetTrainerENet_23_1_s19_warmstart_4c")
    parser.add_argument("--dataset-name", default="Dataset509_ARCADE_1x1_4c")
    parser.add_argument("--plans-name", default="nnUNetPlans")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pth")
    parser.add_argument("--n-batches", type=int, default=16, help="Real preprocessed images to average the trace estimate over.")
    parser.add_argument("--n-probes", type=int, default=10, help="Rademacher probes per batch (Hutchinson's method).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-file", type=Path, default=Path("compression/hawq/sensitivity_23_1.json"))
    args = parser.parse_args()

    checkpoint_path = (
        NNUNET_RESULTS / args.dataset_name / f"{args.net_name}__{args.plans_name}__{args.configuration}"
        / f"fold_{args.fold}" / args.checkpoint_name
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    report = run_sensitivity(checkpoint_path, args.dataset_name, args.n_batches, args.n_probes, args.device, args.seed)

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out_file}")
    for stage, entry in report.items():
        print(f"  {stage}: trace_w={entry['trace_w']:.4e} trace_a={entry['trace_a']:.4e} "
              f"n_weight_params={entry['n_weight_params']}")


if __name__ == "__main__":
    main()
