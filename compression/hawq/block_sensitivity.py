"""Per-BOTTLENECK-BLOCK HAWQ-style sensitivity estimator -- same Hutchinson-
trace Hessian estimation + per-candidate-bit quantization error as
sensitivity.py, but grouped by INDIVIDUAL ENet bottleneck block (see
block_utils.enumerate_blocks) instead of sensitivity.py's static 5-way
stage grouping (initial/stage1/context/stage4/stage5). Produces a
block_sensitivity_<config>.json that ilp_search.py can consume directly
(same {"trace_w", "trace_a", "delta_w", "delta_a", "sensitivity_w",
"sensitivity_a", ...} schema per entry, just many more entries -- one per
block instead of one per stage) to assign every bottleneck its own
independent W/A bit-width, rather than one shared choice across an entire
stage.

Runs entirely on the plain FP32 ENet (loaded from its own checkpoint) --
NOT on a Quant* model -- to avoid double-backward through Brevitas's
straight-through-estimator fake-quant ops, and because sensitivity is a
property of the loss landscape near the FP32 optimum, independent of which
quantization scheme is later applied (same practice sensitivity.py and
HAWQ's own papers use). See sensitivity.py's own module docstring for the
full method writeup (Hutchinson's trace estimator, fake-quantize
convention, etc.) -- not repeated here, this file only changes the
GROUPING, not the underlying math.

Usage:
    python compression/hawq/block_sensitivity.py \\
        --config config_26_5_w24 \\
        --net-name nnUNetTrainerENet_26_5_w24 \\
        --checkpoint-name checkpoint_best.pth \\
        --n-batches 16 --n-probes 10 \\
        --out-file compression/hawq/artifacts/block_sensitivity_26_5_w24.json
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
# module globals, not this file's -- it reads a bare `CANDIDATE_BITS` name
# resolved in sensitivity.py's __globals__, which normally only gets
# populated by that file's own load_config(). Since this file never calls
# that, inject it directly (same value, every config_*.py agrees on it).
_sensitivity.CANDIDATE_BITS = CANDIDATE_BITS


def load_config(config_module: str) -> None:
    """Same pattern as sensitivity.py's own loader -- injects the named
    config_*.py's constants (CHANNELS, BOTTLENECKS_PER_STAGE, ...) into this
    module's globals."""
    cfg = importlib.import_module(config_module)
    globals().update({k: v for k, v in vars(cfg).items() if not k.startswith("_")})


def build_fp32_model(checkpoint_path: Path) -> ENet:
    # OPTIONAL config globals -- see sensitivity.py's own build_fp32_model
    # for the full rationale (not repeated here): every pre-S8.2 config_*.py
    # never defines these (PReLU + plain projected bottlenecks), so .get()
    # falls back to ENet.py's own defaults, unchanged for them.
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


def run_block_sensitivity(
    checkpoint_path: Path, dataset_name: str, n_batches: int, n_probes: int, device: str, seed: int,
) -> dict:
    model = build_fp32_model(checkpoint_path).to(device)
    model.eval()  # BN uses running stats; dropout off -- sensitivity should reflect inference-time behavior

    blocks = enumerate_blocks(model)
    block_names = list(blocks.keys())
    print(f"Enumerated {len(block_names)} bottleneck blocks: {block_names}")

    loss_fn = DC_and_CE_loss(
        {"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False}, {},
        weight_ce=1, weight_dice=1, ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss,
    )
    weight_targets_by_block = block_weight_targets(blocks)

    captured: dict[str, torch.Tensor] = {}

    def make_hook(block_name: str):
        def hook(_module, _inputs, output):
            # DownsamplingBottleneck (down1/down2) returns (x, indices, size),
            # not a plain tensor -- same tuple-output quirk finn_stage_costs.py's
            # own MaxPool2d(return_indices=True) hook already unwraps. Every
            # other block returns a plain tensor.
            captured[block_name] = output[0] if isinstance(output, tuple) else output
        return hook

    handles = [module.register_forward_hook(make_hook(name)) for name, module in blocks.items()]

    batches = load_real_batches(dataset_name, n_batches, seed=seed)
    print(f"Loaded {len(batches)} real (image, seg) batches from {dataset_name}.")

    weight_trace_sum = {b: {n: 0.0 for n in weight_targets_by_block[b]} for b in block_names}
    weight_numel = {b: {n: t.numel() for n, t in weight_targets_by_block[b].items()} for b in block_names}
    act_trace_sum = {b: 0.0 for b in block_names}
    act_numel_sum = {b: 0 for b in block_names}
    weight_delta_sum = {b: {bit: 0.0 for bit in CANDIDATE_BITS} for b in block_names}
    weight_numel_total = {b: 0 for b in block_names}
    act_delta_sum = {b: {bit: 0.0 for bit in CANDIDATE_BITS} for b in block_names}
    act_numel_total = {b: 0 for b in block_names}

    n_used = 0
    for batch_i, (img, seg) in enumerate(batches):
        img, seg = img.to(device), seg.to(device)
        captured.clear()
        try:
            out = model(img)
        except RuntimeError as error:
            # Same known real-data shape quirk sensitivity.py's own run_sensitivity
            # already documents and skips (H/W not evenly divisible by ENet's
            # stride-8 downsample factor) -- a representative sample is enough.
            print(f"  [skip] batch {batch_i + 1}/{len(batches)} (shape {tuple(img.shape)}) failed forward pass: {error}")
            continue
        loss = loss_fn(out, seg)
        n_used += 1

        for block_name in block_names:
            wt = weight_targets_by_block[block_name]
            if wt:
                traces = hutchinson_traces(loss, wt, n_probes)
                for n, tr in traces.items():
                    weight_trace_sum[block_name][n] += tr
            act_traces = hutchinson_traces(loss, {block_name: captured[block_name]}, n_probes)
            act_trace_sum[block_name] += act_traces[block_name]
            act_numel_sum[block_name] += captured[block_name].numel()

        with torch.no_grad():
            for block_name, wt in weight_targets_by_block.items():
                for tensor in wt.values():
                    deltas = quantization_deltas(tensor)
                    for b, d in deltas.items():
                        weight_delta_sum[block_name][b] += d
                    weight_numel_total[block_name] += tensor.numel()
            for block_name in block_names:
                deltas = quantization_deltas(captured[block_name])
                for b, d in deltas.items():
                    act_delta_sum[block_name][b] += d
                act_numel_total[block_name] += captured[block_name].numel()

        print(f"batch {batch_i + 1}/{len(batches)} done.")

    for h in handles:
        h.remove()

    if n_used == 0:
        raise RuntimeError("Every batch failed the forward pass -- can't estimate sensitivity at all.")
    print(f"Used {n_used}/{len(batches)} batches (some may have been skipped for shape reasons, see above).")

    report = {}
    for block_name in block_names:
        total_w_trace = sum(weight_trace_sum[block_name].values()) / n_used
        total_w_numel = sum(weight_numel[block_name].values())
        trace_w = total_w_trace / max(total_w_numel, 1)
        trace_a = (act_trace_sum[block_name] / n_used) / max(act_numel_sum[block_name] / n_used, 1)

        delta_w = {b: weight_delta_sum[block_name][b] / n_used for b in CANDIDATE_BITS}
        delta_a = {b: act_delta_sum[block_name][b] / n_used for b in CANDIDATE_BITS}

        report[block_name] = {
            "trace_w": trace_w,
            "trace_a": trace_a,
            "delta_w": {str(b): delta_w[b] for b in CANDIDATE_BITS},
            "delta_a": {str(b): delta_a[b] for b in CANDIDATE_BITS},
            "sensitivity_w": {str(b): trace_w * delta_w[b] for b in CANDIDATE_BITS},
            "sensitivity_a": {str(b): trace_a * delta_a[b] for b in CANDIDATE_BITS},
            "n_weight_tensors": len(weight_targets_by_block[block_name]),
            "n_weight_params": total_w_numel,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_23_1",
                         help="Which compression/hawq/config_*.py to load -- e.g. config_23_1 or config_26_5_w24.")
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
                              "same override convention joint_bits_folding_ilp.py's own --candidate-bits uses. "
                              "Needed whenever a downstream ILP run wants a bit-width this config's own default "
                              "CANDIDATE_BITS never measured (e.g. config_12_separable_dense_relu's own default "
                              "is (2,4,8) -- asking a joint ILP for --candidate-bits 4,6,8 fails with "
                              "KeyError: '6' unless this file is re-run with 6 included first).")
    parser.add_argument("--out-file", type=Path, default=None,
                         help="Defaults to compression/hawq/block_sensitivity_<config suffix>.json.")
    args = parser.parse_args()

    load_config(args.config)  # populates CHANNELS/BOTTLENECKS_PER_STAGE/NET_NAME/... as module globals
    if args.candidate_bits is not None:
        global CANDIDATE_BITS
        CANDIDATE_BITS = tuple(sorted(int(b) for b in args.candidate_bits.split(",")))
        print(f"Overriding CANDIDATE_BITS to {CANDIDATE_BITS} (from --candidate-bits).")
    # Re-sync: load_config() (or the --candidate-bits override above) just
    # rebound this module's OWN CANDIDATE_BITS global, but quantization_deltas()
    # lives in sensitivity.py and reads ITS module's CANDIDATE_BITS -- the
    # one-time injection at import time (module-level, above) used this
    # file's stale pre-load_config() default (2,4,8,16), not the config (or
    # override) actually in effect here. Without this, any CANDIDATE_BITS !=
    # (2,4,8,16) raises KeyError inside run_block_sensitivity's
    # weight_delta_sum/act_delta_sum accumulation.
    _sensitivity.CANDIDATE_BITS = CANDIDATE_BITS
    net_name = args.net_name or NET_NAME
    out_file = args.out_file or Path(f"compression/hawq/artifacts/block_sensitivity_{args.config.removeprefix('config_')}.json")

    checkpoint_path = (
        NNUNET_RESULTS / args.dataset_name / f"{net_name}__{args.plans_name}__{args.configuration}"
        / f"fold_{args.fold}" / args.checkpoint_name
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    report = run_block_sensitivity(checkpoint_path, args.dataset_name, args.n_batches, args.n_probes, args.device, args.seed)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_file}")
    for block_name, entry in report.items():
        print(f"  {block_name}: trace_w={entry['trace_w']:.4e} trace_a={entry['trace_a']:.4e} "
              f"n_weight_params={entry['n_weight_params']}")


if __name__ == "__main__":
    main()
