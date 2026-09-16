"""Per-INDIVIDUAL-CONV-LAYER HAWQ-style sensitivity estimator -- same
Hutchinson-trace Hessian estimation + per-candidate-bit quantization error as
block_sensitivity.py/sensitivity.py, but reported per individual conv layer
(see block_utils.block_weight_targets) instead of block_sensitivity.py's
per-BOTTLENECK-BLOCK aggregation. Produces a layer_sensitivity_<config>.json
with one entry per real Conv2d/ConvTranspose2d layer (101 for S12, vs. 29
blocks in block_sensitivity_*.json); each entry carries a "stage" field
naming its owning block for traceability back to the coarser artifact.

block_utils.block_weight_targets already collects one weight tensor per
individual conv layer internally -- block_sensitivity.py just aggregates
those back down to one number per block at its own report-writing step (see
that file's own run_block_sensitivity: trace_w is a params-weighted MEAN
across a block's own layers, delta_w/delta_a are plain SUMs -- two DIFFERENT
reductions). This file skips that aggregation and reports every layer
independently instead. Activation sensitivity, by contrast, is genuinely new
here: block_sensitivity.py only ever hooked one boundary tensor per whole
block; this file hooks each individual conv layer's own output.

Runs entirely on the plain FP32 ENet (loaded from its own checkpoint) -- NOT
on a Quant* model -- same rationale as block_sensitivity.py/sensitivity.py
(sensitivity is a property of the FP32 loss landscape near the optimum,
independent of which quantization scheme is later applied).

hutchinson_traces is called ONCE per batch across ALL layers' weight
tensors (not once per layer), and once per batch across ALL layers'
captured activations -- rather than block_sensitivity.py's one-call-per-block
loop: cross terms between independently-drawn Rademacher probes on
different tensors vanish in expectation, so batching every tensor into a
single call is both mathematically equivalent and, since it shares one
first-order backward graph across everything, strictly cheaper.

Usage:
    python compression/hawq/layer_sensitivity.py \\
        --config config_12_separable_dense_relu \\
        --net-name nnUNetTrainerENet_12_separable_dense_relu \\
        --checkpoint-name checkpoint_best.pth \\
        --n-batches 16 --n-probes 10 \\
        --out-file compression/hawq/artifacts/layer_sensitivity_12_separable_dense_relu.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss  # noqa: E402
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_utils import block_weight_targets, enumerate_blocks  # noqa: E402
import sensitivity as _sensitivity  # noqa: E402 -- reuse the exact same primitives, not reimplemented
from sensitivity import (  # noqa: E402
    fake_quantize_symmetric, hutchinson_traces, load_real_batches, quantization_deltas, rademacher_like,
)

NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"

CANDIDATE_BITS = (2, 4, 8, 16)
# quantization_deltas (imported above) is a closure over sensitivity.py's OWN
# module globals, not this file's -- see block_sensitivity.py's identical
# comment. Inject this file's stale default now (re-injected again in main()
# after load_config() actually loads the real config -- see that call site).
_sensitivity.CANDIDATE_BITS = CANDIDATE_BITS


def load_config(config_module: str) -> None:
    """Same pattern as block_sensitivity.py/sensitivity.py's own loader --
    injects the named config_*.py's constants into this module's globals."""
    cfg = importlib.import_module(config_module)
    globals().update({k: v for k, v in vars(cfg).items() if not k.startswith("_")})


def build_fp32_model(checkpoint_path: Path) -> ENet:
    # OPTIONAL config globals -- see sensitivity.py's own build_fp32_model for
    # the full rationale (not repeated here).
    model = ENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        decoder_type=DECODER_TYPE, use_asymmetric=USE_ASYMMETRIC, context_pattern=CONTEXT_PATTERN,
        separable_dilated=SEPARABLE_DILATED, use_prelu=globals().get("USE_PRELU", True), prelu_variant=PRELU_VARIANT,
        use_dsc=globals().get("USE_DSC", False), dsc_no_projection=globals().get("DSC_NO_PROJECTION", False),
        dsc_no_projection_context_only=globals().get("DSC_NO_PROJECTION_CONTEXT_ONLY", False),
        reg_bookend_dsc=globals().get("REG_BOOKEND_DSC", False),
        dsc_separable=globals().get("DSC_SEPARABLE", False),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["network_weights"], strict=True)
    return model


def run_layer_sensitivity(
    checkpoint_path: Path, dataset_name: str, n_batches: int, n_probes: int, device: str, seed: int,
) -> dict:
    model = build_fp32_model(checkpoint_path).to(device)
    model.eval()  # BN uses running stats; dropout off -- sensitivity should reflect inference-time behavior

    blocks = enumerate_blocks(model)
    weight_targets_by_block = block_weight_targets(blocks)

    flat_weight_targets: dict[str, torch.Tensor] = {}
    layer_to_block: dict[str, str] = {}
    for block_name, tensors in weight_targets_by_block.items():
        for layer_name, tensor in tensors.items():
            flat_weight_targets[layer_name] = tensor
            layer_to_block[layer_name] = block_name
    layer_names = list(flat_weight_targets)
    print(f"Enumerated {len(layer_names)} individual conv layers across {len(blocks)} blocks.")

    loss_fn = DC_and_CE_loss(
        {"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False}, {},
        weight_ce=1, weight_dice=1, ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss,
    )

    captured: dict[str, torch.Tensor] = {}

    def make_hook(layer_name: str):
        def hook(_module, _inputs, output):
            # Tuple-output guard kept for defensive parity with every other
            # hook in this codebase (block_sensitivity.py/finn_block_costs.py)
            # -- no individual Conv2d/ConvTranspose2d hooked here actually
            # returns a tuple itself (that quirk belongs to a DownsamplingBottleneck's
            # own forward, one level up, not to a single conv).
            captured[layer_name] = output[0] if isinstance(output, tuple) else output
        return hook

    named_modules = dict(model.named_modules())
    handles = [named_modules[name].register_forward_hook(make_hook(name)) for name in layer_names]
    assert len(handles) == len(layer_names)

    batches = load_real_batches(dataset_name, n_batches, seed=seed)
    print(f"Loaded {len(batches)} real (image, seg) batches from {dataset_name}.")

    weight_trace_sum = {n: 0.0 for n in layer_names}
    weight_numel = {n: t.numel() for n, t in flat_weight_targets.items()}
    act_trace_sum = {n: 0.0 for n in layer_names}
    act_numel_sum = {n: 0 for n in layer_names}
    weight_delta_sum = {n: {bit: 0.0 for bit in CANDIDATE_BITS} for n in layer_names}
    act_delta_sum = {n: {bit: 0.0 for bit in CANDIDATE_BITS} for n in layer_names}

    n_used = 0
    for batch_i, (img, seg) in enumerate(batches):
        img, seg = img.to(device), seg.to(device)
        captured.clear()
        try:
            out = model(img)
        except RuntimeError as error:
            # Same known real-data shape quirk sensitivity.py's/block_sensitivity.py's
            # own run functions already document and skip (H/W not evenly
            # divisible by ENet's stride-8 downsample factor). A representative
            # sample is enough.
            print(f"  [skip] batch {batch_i + 1}/{len(batches)} (shape {tuple(img.shape)}) failed forward pass: {error}")
            continue
        loss = loss_fn(out, seg)
        n_used += 1

        # ONE hutchinson_traces call across ALL layers' weight tensors, and
        # ONE across all layers' captured activations -- see module docstring
        # for why this is both correct and cheaper than block_sensitivity.py's
        # one-call-per-block loop.
        w_traces = hutchinson_traces(loss, flat_weight_targets, n_probes)
        for n, tr in w_traces.items():
            weight_trace_sum[n] += tr
        a_traces = hutchinson_traces(loss, captured, n_probes)
        for n, tr in a_traces.items():
            act_trace_sum[n] += tr
            act_numel_sum[n] += captured[n].numel()

        with torch.no_grad():
            for n, tensor in flat_weight_targets.items():
                deltas = quantization_deltas(tensor)
                for b, d in deltas.items():
                    weight_delta_sum[n][b] += d
            for n in layer_names:
                deltas = quantization_deltas(captured[n])
                for b, d in deltas.items():
                    act_delta_sum[n][b] += d

        print(f"batch {batch_i + 1}/{len(batches)} done.")

    for h in handles:
        h.remove()

    if n_used == 0:
        raise RuntimeError("Every batch failed the forward pass -- can't estimate sensitivity at all.")
    print(f"Used {n_used}/{len(batches)} batches (some may have been skipped for shape reasons, see above).")

    report = {}
    for layer_name in layer_names:
        trace_w = (weight_trace_sum[layer_name] / n_used) / max(weight_numel[layer_name], 1)
        trace_a = (act_trace_sum[layer_name] / n_used) / max(act_numel_sum[layer_name] / n_used, 1)

        delta_w = {b: weight_delta_sum[layer_name][b] / n_used for b in CANDIDATE_BITS}
        delta_a = {b: act_delta_sum[layer_name][b] / n_used for b in CANDIDATE_BITS}

        report[layer_name] = {
            "stage": layer_to_block[layer_name],
            "trace_w": trace_w,
            "trace_a": trace_a,
            "delta_w": {str(b): delta_w[b] for b in CANDIDATE_BITS},
            "delta_a": {str(b): delta_a[b] for b in CANDIDATE_BITS},
            "sensitivity_w": {str(b): trace_w * delta_w[b] for b in CANDIDATE_BITS},
            "sensitivity_a": {str(b): trace_a * delta_a[b] for b in CANDIDATE_BITS},
            "n_weight_params": weight_numel[layer_name],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_23_1",
                         help="Which compression/hawq/config_*.py to load -- e.g. config_12_separable_dense_relu.")
    parser.add_argument("--net-name", default=None, help="Defaults to the loaded config's own NET_NAME.")
    parser.add_argument("--dataset-name", default="Dataset509_ARCADE_1x1_4c")
    parser.add_argument("--plans-name", default="nnUNetPlans")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pth")
    parser.add_argument("--n-batches", type=int, default=16, help="Real preprocessed images to average the trace estimate over.")
    parser.add_argument("--n-probes", type=int, default=10, help="Rademacher probes per batch (Hutchinson's method).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--candidate-bits", type=str, default=None,
                         help="Comma-separated override for the module-level CANDIDATE_BITS (e.g. '2,4,6,8') -- "
                              "same override convention block_sensitivity.py's/joint_bits_folding_ilp.py's own "
                              "--candidate-bits use. Needed whenever a downstream per-layer ILP run wants a "
                              "bit-width this config's own default CANDIDATE_BITS never measured.")
    parser.add_argument("--out-file", type=Path, default=None,
                         help="Defaults to compression/hawq/artifacts/layer_sensitivity_<config suffix>.json.")
    args = parser.parse_args()

    load_config(args.config)  # populates CHANNELS/BOTTLENECKS_PER_STAGE/NET_NAME/... as module globals
    if args.candidate_bits is not None:
        global CANDIDATE_BITS
        CANDIDATE_BITS = tuple(sorted(int(b) for b in args.candidate_bits.split(",")))
        print(f"Overriding CANDIDATE_BITS to {CANDIDATE_BITS} (from --candidate-bits).")
    # Re-sync -- see block_sensitivity.py's own identical comment for why this
    # second injection (after load_config, or the --candidate-bits override
    # above, not just the module-load-time one above) is required: without
    # it, any CANDIDATE_BITS != this file's stale (2,4,8,16) default raises
    # KeyError inside quantization_deltas.
    _sensitivity.CANDIDATE_BITS = CANDIDATE_BITS
    net_name = args.net_name or NET_NAME
    out_file = args.out_file or Path(f"compression/hawq/artifacts/layer_sensitivity_{args.config.removeprefix('config_')}.json")

    checkpoint_path = (
        NNUNET_RESULTS / args.dataset_name / f"{net_name}__{args.plans_name}__{args.configuration}"
        / f"fold_{args.fold}" / args.checkpoint_name
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    report = run_layer_sensitivity(checkpoint_path, args.dataset_name, args.n_batches, args.n_probes, args.device, args.seed)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_file}")
    for layer_name, entry in report.items():
        print(f"  {layer_name} ({entry['stage']}): trace_w={entry['trace_w']:.4e} trace_a={entry['trace_a']:.4e} "
              f"n_weight_params={entry['n_weight_params']}")


if __name__ == "__main__":
    main()
