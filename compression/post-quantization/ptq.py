"""Post-training quantization (PTQ) for QuantENet: calibrate a Brevitas-
quantized mirror of an already-trained FP32 ENet checkpoint, with NO
retraining -- transfers the FP32 weights directly (name+shape matched),
then runs Brevitas's calibration_mode over real preprocessed images to set
every quantizer's scale/zero-point, and saves the result as a standard
nnU-Net checkpoint folder that compression/collect_results.py can evaluate
exactly like any other config (real inference, real dice, no special-casing
needed there).

Weight transfer uses a direct strict=False load, NOT nnU-Net's own generic
enet/nnunetv2/run/load_pretrained_weights.py (built for FP32<->FP32
transfers): Brevitas's quantizer scale parameters (e.g.
`*.scaling_impl.value`) are real nn.Parameters (show up in
named_parameters()) but are NOT written by state_dict() on a fresh,
uncalibrated model -- yet load_state_dict()'s strict mode still demands
them, an asymmetry confirmed on a completely bare QuantENet with no
relation to this file's own code. load_pretrained_weights' generic
"build a full model_dict then load strict=True" approach trips exactly this
(confirmed empirically -- RuntimeError, "Missing key(s)" listing every
scaling_impl.value path). strict=False sidesteps it directly: every real
conv/BN weight still transfers (confirmed 482/482 keys matched by name+
shape on S8-ReLU's architecture), and the naturally-unprovided quantizer
buffers are exactly what calibration_mode below is about to populate
anyway.

Answers a different question than nnUNetTrainerENetQuant.py's QAT path
(compression/slurm/stage_14/15_*_quant_*.job): PTQ measures how much
quantizing an ALREADY-TRAINED model hurts, with zero retraining cost (a few
calibration forward passes, no backward pass at all); QAT measures how well
the network can learn to compensate for quantization from scratch (a full
150-epoch retrain). Different costs, different comparisons -- both are
useful, neither replaces the other.

Reusable across any trained FP32 checkpoint + target bit-width, not
hardcoded to a specific config -- every architecture flag is passed
explicitly on the command line, mirroring collect_results.py's own CLI
surface (same flag names, same semantics) rather than re-parsing them out
of results.csv's ops_flags string (whose leaky_slope_map value embeds its
own commas/equals-signs, unsafe to comma-split generically).

Usage (S8-ReLU @ INT4):
    python compression/post-quantization/ptq.py \
        --source-net-name nnUNetTrainerENet_8_2_relu \
        --out-net-name nnUNetTrainerENetQuant_8_2_relu_ptq_int4 \
        --channels 4,16,32,16,4 --bottlenecks 4,11,11,2,1 --decoder-type upsample_conv \
        --use-asymmetric 0 --context-pattern dense_dilation_reg_interleaved --dsc-no-projection 1 \
        --quant-bits 4

Usage (S13 frozen-leaky @ INT8, with its own per-block slope map):
    python compression/post-quantization/ptq.py \
        --source-net-name nnUNetTrainerENet_13_separable_dense_nonneg_block_leaky_frozen \
        --out-net-name nnUNetTrainerENetQuant_13_leaky_frozen_ptq_int8 \
        --channels 4,16,32,16,4 --bottlenecks 4,8,8,2,1 --decoder-type upsample_conv \
        --use-asymmetric 0 --context-pattern dense_dilation --separable-dilated 1 \
        --leaky-slope-map-file compression/post-quantization/slope_maps/13_separable_dense_nonneg_block_leaky_frozen.json \
        --quant-bits 8

Then evaluate normally:
    python compression/collect_results.py --net-name nnUNetTrainerENetQuant_8_2_relu_ptq_int4 \
        --stage experiment_s8relu_ptq_int4 --channels 4,16,32,16,4 --bottlenecks 4,11,11,2,1 \
        --decoder-type upsample_conv --use-asymmetric 0 --context-pattern dense_dilation_reg_interleaved \
        --dsc-no-projection 1 --quant-bits 4 --trainer-class nnUNetTrainerENetQuant
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.QuantENet import QuantENet  # noqa: E402

from brevitas.graph.calibrate import calibration_mode  # noqa: E402

NNUNET_PREPROCESSED = REPO_ROOT / "data" / "nnUNet_preprocessed"
NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"


def parse_tuple5(value: str, name: str) -> tuple[int, ...]:
    parts = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if len(parts) != 5:
        raise ValueError(f"--{name} must have exactly 5 comma-separated integers, got {value!r}.")
    return parts


def load_calibration_batches(
    dataset_name: str, n_images: int, batch_size: int, seed: int = 0,
) -> list[torch.Tensor]:
    """Samples n_images real PREPROCESSED (already z-score-normalized)
    training images at random from
    nnUNet_preprocessed/<dataset>/nnUNetPlans_2d/*.npy -- the exact same
    tensors the real trainer feeds the network, not synthetic/random noise
    (calibration statistics need to reflect the real input distribution).
    Segmentation labels aren't needed -- calibration only runs forward
    passes, never computes a loss.

    Real per-case images have VARYING H/W (not all exactly the 512x512
    training patch size -- confirmed empirically, e.g. one real case is
    508x512) -- can't be torch.stack'd into a shared batch_size-sized
    tensor without cropping/padding. ENet's own forward pass is fully
    convolutional and threads explicit per-call spatial sizes through its
    up4/up5 stages (see ENet.py/QuantENet.py's own forward()), so it
    tolerates arbitrary H/W directly -- simplest robust fix is just batch
    size 1 (each image its own calibration forward pass) rather than
    complicating this with resize/crop logic calibration doesn't need.
    `batch_size` is accepted for interface symmetry with a future fixed-
    size calibration corpus, but is a no-op today (see assertion below)."""
    if batch_size != 1:
        raise NotImplementedError(
            "calibration_batch_size > 1 isn't supported yet -- real preprocessed images have varying "
            "H/W and can't be torch.stack'd without cropping/padding (not implemented). Use 1."
        )
    preprocessed_dir = NNUNET_PREPROCESSED / dataset_name / "nnUNetPlans_2d"
    image_files = sorted(p for p in preprocessed_dir.glob("*.npy") if not p.name.endswith("_seg.npy"))
    if not image_files:
        raise FileNotFoundError(f"No preprocessed .npy images found under {preprocessed_dir}")
    rng = random.Random(seed)
    sampled = rng.sample(image_files, k=min(n_images, len(image_files)))
    return [torch.from_numpy(np.load(p)).float() for p in sampled]  # each already (1, 1, H, W)


def build_quant_model(
    in_channels: int, out_channels: int, channels: tuple[int, ...], bottlenecks: tuple[int, ...],
    decoder_type: str, use_dilated: bool, use_asymmetric: bool, use_strided: bool, use_dsc: bool,
    quant_bits: int, context_pattern: str, dsc_no_projection: bool, dsc_no_projection_context_only: bool,
    separable_dilated: bool, leaky_slope_map: dict | None,
) -> QuantENet:
    return QuantENet(
        in_channels=in_channels, out_channels=out_channels, channels=channels,
        bottlenecks_per_stage=bottlenecks, decoder_type=decoder_type,
        use_dilated=use_dilated, use_asymmetric=use_asymmetric, use_strided=use_strided, use_dsc=use_dsc,
        weight_bit_width=quant_bits, act_bit_width=quant_bits,
        context_pattern=context_pattern, dsc_no_projection=dsc_no_projection,
        dsc_no_projection_context_only=dsc_no_projection_context_only,
        separable_dilated=separable_dilated, leaky_slope_map=leaky_slope_map,
    )


def transfer_fp32_weights(quant_model: QuantENet, fp32_checkpoint_path: Path) -> dict:
    """Direct strict=False name+shape-matched transfer -- see module
    docstring for why this doesn't go through nnU-Net's generic
    load_pretrained_weights. Returns a small stats dict for logging."""
    checkpoint = torch.load(fp32_checkpoint_path, map_location="cpu", weights_only=False)
    source_state_dict = checkpoint["network_weights"]
    model_state_dict = quant_model.state_dict()
    transferable = {
        key: value for key, value in source_state_dict.items()
        if key in model_state_dict and model_state_dict[key].shape == value.shape
    }
    n_shape_mismatch = sum(
        1 for key, value in source_state_dict.items()
        if key in model_state_dict and model_state_dict[key].shape != value.shape
    )
    n_source_only = len(source_state_dict) - len(transferable) - n_shape_mismatch
    missing, unexpected = quant_model.load_state_dict(transferable, strict=False)
    assert not unexpected, f"unexpected keys after strict=False load (should be impossible, transferable is a subset of model_state_dict): {unexpected}"
    return {
        "n_transferred": len(transferable),
        "n_model_keys": len(model_state_dict),
        "n_shape_mismatch": n_shape_mismatch,
        "n_source_only": n_source_only,
        "n_missing_after_load": len(missing),
    }


def calibrate(quant_model: QuantENet, calibration_batches: list[torch.Tensor], device: str) -> int:
    """Runs Brevitas's calibration_mode: forward-only passes (no backward,
    no optimizer) that let every quantizer's observer collect real
    activation statistics, then finalize each into a fixed scale/zero-point
    on __exit__. This is the entire "training" this script does -- no
    gradient updates to the underlying FP32 weights at all, which is
    exactly what makes this POST-training (not quantization-aware
    training).

    Some real preprocessed images have odd H/W (not evenly divisible by
    the network's total stride-2 downsample factor of 8) and trip a shape
    mismatch in the upsample_conv decoder's residual add (a real
    architectural sensitivity independent of quantization -- see ENet.py's
    own UpsamplingBottleneck docstring on patch-size alignment). Calibration
    only needs a reasonably representative SAMPLE of activation statistics,
    not every single image to succeed, so a failing image is skipped (with
    a warning) rather than aborting the whole run. Returns the number of
    images actually used."""
    quant_model.to(device)
    quant_model.train()  # calibration_mode needs is_training=True internally to collect stats
    n_used = 0
    with torch.no_grad(), calibration_mode(quant_model):
        for batch in calibration_batches:
            try:
                quant_model(batch.to(device))
                n_used += 1
            except RuntimeError as error:
                print(f"  [skip] calibration image with shape {tuple(batch.shape)} failed forward pass: {error}")
    quant_model.eval()
    if n_used == 0:
        raise RuntimeError("Every calibration image failed -- can't calibrate quantizer scales at all.")
    return n_used


def save_calibrated_checkpoint(
    quant_model: QuantENet, reference_model_folder: Path, reference_checkpoint_name: str,
    out_net_name: str, dataset_name: str, plans_name: str, configuration: str, fold: int,
) -> Path:
    """Packages the calibrated model into a standard nnU-Net checkpoint
    folder -- same shape collect_results.py/nnUNetv2_predict already know
    how to evaluate, no special-casing needed for a PTQ result vs. a real
    trained one. Copies dataset.json/plans.json/dataset_fingerprint.json
    from the reference (FP32) model folder (architecture-independent
    metadata), and stamps trainer_name=nnUNetTrainerENetQuant (NOT the
    FP32 reference's own nnUNetTrainerENet) so nnUNetv2_predict's
    checkpoint-driven class lookup reconstructs the right architecture.

    network_weights is quant_model.state_dict() MERGED with
    quant_model.named_parameters(remove_duplicate=False) (NOT
    named_buffers() -- confirmed that breaks a different Brevitas-internal
    `_load_from_state_dict` override that expects one specific buffer key
    to be ABSENT, ValueError: "list.remove(x): x not in list" on its own
    migration-style load path). Two separate reasons plain state_dict()
    alone isn't enough:
      1. Brevitas's calibrated `*.scaling_impl.value` quantizer-scale
         parameters are real nn.Parameters (present in named_parameters())
         but excluded from state_dict()'s own export entirely -- confirmed
         on a bare, fresh QuantENet, no relation to this file's own code.
      2. `remove_duplicate=False` specifically (not named_parameters()'s
         own default): separable_dilated's QuantRegularBottleneck
         (ENet.py's RegularBottleneck has the identical pattern) builds
         `self.conv_bn_act = nn.Sequential(self.conv, BN, Act)`, i.e.
         conv_bn_act[0] IS THE SAME OBJECT as self.conv, not a copy --
         PyTorch's default named_parameters()/state_dict() traversal
         dedupes shared submodule objects, silently dropping whichever
         path is visited second (here, everything under
         "*.conv_bn_act.0.*"). state_dict() doesn't dedupe plain
         parameters (confirmed separately, see ENet.py's own equivalent
         FP32 checkpoint-filtering code) so this only bites the
         Brevitas-only scaling_impl.value keys living under the aliased
         path -- exactly the ones state_dict() already excludes for
         reason 1, compounding into "missing even after the first fix"
         until remove_duplicate=False is added too.
    Without both fixes, nnU-Net's inference-time
    `self.network.load_state_dict(params)` (strict=True, predict_from_
    raw_data.py) fails with "Missing key(s)" listing every scaling_impl.
    value path (canonical AND aliased), discarding the calibration this
    script just ran. Confirmed empirically: state_dict() + named_
    parameters(remove_duplicate=False) round-trips through a fresh
    model's strict load_state_dict cleanly, including a separable_dilated
    config."""
    reference_checkpoint_path = reference_model_folder / f"fold_{fold}" / reference_checkpoint_name
    reference_checkpoint = torch.load(reference_checkpoint_path, map_location="cpu", weights_only=False)

    out_model_folder = NNUNET_RESULTS / dataset_name / f"{out_net_name}__{plans_name}__{configuration}"
    out_fold_dir = out_model_folder / f"fold_{fold}"
    out_fold_dir.mkdir(parents=True, exist_ok=True)
    for meta_file in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
        src = reference_model_folder / meta_file
        if src.exists():
            (out_model_folder / meta_file).write_bytes(src.read_bytes())

    network_weights = dict(quant_model.state_dict())
    network_weights.update(dict(quant_model.named_parameters(remove_duplicate=False)))
    new_checkpoint = dict(reference_checkpoint)
    new_checkpoint["network_weights"] = network_weights
    new_checkpoint["trainer_name"] = "nnUNetTrainerENetQuant"
    out_checkpoint_path = out_fold_dir / "checkpoint_best.pth"
    torch.save(new_checkpoint, out_checkpoint_path)
    return out_checkpoint_path


def run_ptq(
    source_net_name: str, out_net_name: str, dataset_name: str, in_channels: int, out_channels: int,
    channels: tuple[int, ...], bottlenecks: tuple[int, ...], decoder_type: str, use_dilated: bool,
    use_asymmetric: bool, use_strided: bool, use_dsc: bool, quant_bits: int, context_pattern: str,
    dsc_no_projection: bool, dsc_no_projection_context_only: bool, separable_dilated: bool,
    leaky_slope_map: dict | None, plans_name: str, configuration: str, fold: int,
    source_checkpoint_name: str, n_calibration_images: int, calibration_batch_size: int,
    calibration_seed: int, device: str,
) -> Path:
    """The full reusable pipeline: build -> transfer FP32 weights -> calibrate -> save.
    Returns the path to the new checkpoint, ready for collect_results.py."""
    quant_model = build_quant_model(
        in_channels, out_channels, channels, bottlenecks, decoder_type, use_dilated, use_asymmetric,
        use_strided, use_dsc, quant_bits, context_pattern, dsc_no_projection, dsc_no_projection_context_only,
        separable_dilated, leaky_slope_map,
    )

    source_model_folder = NNUNET_RESULTS / dataset_name / f"{source_net_name}__{plans_name}__{configuration}"
    source_checkpoint_path = source_model_folder / f"fold_{fold}" / source_checkpoint_name
    if not source_checkpoint_path.exists():
        raise FileNotFoundError(f"Source FP32 checkpoint not found: {source_checkpoint_path}")
    transfer_stats = transfer_fp32_weights(quant_model, source_checkpoint_path)
    print(f"Weight transfer from {source_checkpoint_path}: {transfer_stats}")

    calibration_batches = load_calibration_batches(
        dataset_name, n_calibration_images, calibration_batch_size, seed=calibration_seed,
    )
    print(f"Calibrating on up to {sum(b.shape[0] for b in calibration_batches)} real preprocessed images...")
    n_used = calibrate(quant_model, calibration_batches, device)
    print(f"Calibration used {n_used}/{len(calibration_batches)} images (some may have been skipped for shape reasons, see above).")

    out_checkpoint_path = save_calibrated_checkpoint(
        quant_model, source_model_folder, source_checkpoint_name, out_net_name,
        dataset_name, plans_name, configuration, fold,
    )
    print(f"Saved calibrated PTQ checkpoint: {out_checkpoint_path}")
    return out_checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-net-name", required=True, help="config_name of the already-trained FP32 ENet checkpoint to quantize (its results.csv net-name).")
    parser.add_argument("--out-net-name", required=True, help="net-name for the new calibrated PTQ checkpoint -- pass this to collect_results.py --net-name afterward.")
    parser.add_argument("--dataset-name", default="Dataset509_ARCADE_1x1_4c")
    parser.add_argument("--channels", required=True, type=lambda v: parse_tuple5(v, "channels"))
    parser.add_argument("--bottlenecks", default="4,8,8,2,1", type=lambda v: parse_tuple5(v, "bottlenecks"))
    parser.add_argument("--decoder-type", default="upsample_conv", choices=["max_unpool", "upsample_conv"])
    parser.add_argument("--use-dilated", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-asymmetric", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-strided", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-dsc", type=int, default=0, choices=[0, 1])
    parser.add_argument("--context-pattern", default="default", choices=["default", "dense_dilation", "dense_dilation_reg_interleaved", "dense_dilation_reg_interleaved_double_mid"])
    parser.add_argument("--dsc-no-projection", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dsc-no-projection-context-only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--separable-dilated", type=int, default=0, choices=[0, 1])
    parser.add_argument("--leaky-slope-map-file", default=None, help="JSON file of {block_name: slope} -- see ENet.py's collect_prelu_block_means for the convention. Builds QuantDecomposedLeakyAct at the mapped blocks instead of plain QuantReLU.")
    parser.add_argument("--quant-bits", type=int, required=True, help="Homogeneous weight+activation bit-width (e.g. 4 or 8).")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--plans-name", default="nnUNetPlans")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--source-checkpoint-name", default="checkpoint_best.pth")
    parser.add_argument("--n-calibration-images", type=int, default=64, help="How many real preprocessed training images to calibrate quantizer scales on.")
    parser.add_argument("--calibration-batch-size", type=int, default=1, help="Must be 1 -- see load_calibration_batches's docstring (real images have varying H/W).")
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    leaky_slope_map = None
    if args.leaky_slope_map_file:
        with open(args.leaky_slope_map_file) as f:
            leaky_slope_map = json.load(f)

    run_ptq(
        source_net_name=args.source_net_name,
        out_net_name=args.out_net_name,
        dataset_name=args.dataset_name,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        channels=args.channels,
        bottlenecks=args.bottlenecks,
        decoder_type=args.decoder_type,
        use_dilated=bool(args.use_dilated),
        use_asymmetric=bool(args.use_asymmetric),
        use_strided=bool(args.use_strided),
        use_dsc=bool(args.use_dsc),
        quant_bits=args.quant_bits,
        context_pattern=args.context_pattern,
        dsc_no_projection=bool(args.dsc_no_projection),
        dsc_no_projection_context_only=bool(args.dsc_no_projection_context_only),
        separable_dilated=bool(args.separable_dilated),
        leaky_slope_map=leaky_slope_map,
        plans_name=args.plans_name,
        configuration=args.configuration,
        fold=args.fold,
        source_checkpoint_name=args.source_checkpoint_name,
        n_calibration_images=args.n_calibration_images,
        calibration_batch_size=args.calibration_batch_size,
        calibration_seed=args.calibration_seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
