"""PTQ + calibration of the PReLU LayerQuantEnetFINN (every act site a QuantPReLU) from an FP32 PReLU checkpoint -- the PReLU sibling of
nas/post-quantization/calibrate_12_dense_relu_bilinear_upsample_256_perlayer.py --model-class finn (reuses its image loader, calibration loop and fp16-safe scale clamp), saving the INTn checkpoint that is
both the PTQ result and the warm start (ENET_PRETRAINED_CHECKPOINT) of the QAT job. The FP32 PReLU alphas transfer by key through from_pretrained.

    python report/hw/activation/calibrate_prelu.py --source-net-name nnUNetTrainerENet_12_dense_prelu_all_warmstart150ep \\
        --out-net-name nnUNetTrainerLayerQuantEnetFINN_12_dense_prelu_all_warmstart150ep_bilinear_uniform_int6_calibrated \\
        --layer-bits-file report/hw/activation/bits/layer_bits_SITES_uniform_int6.json --n-calibration-images 1200 --pad-to-multiple 8
    # encoder-only PReLU source (4_8_dense_dilation): add --no-decoder-prelu
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "enet"))
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_calibrate_relu", REPO_ROOT / "nas" / "post-quantization" / "calibrate_12_dense_relu_bilinear_upsample_256_perlayer.py")
_relu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_relu)

TRAINER_NAME = "nnUNetTrainerLayerQuantEnetFINN_12_dense_prelu_bilinear_upsample_perlayer"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-net-name", required=True)
    p.add_argument("--source-checkpoint-name", default="checkpoint_best.pth")
    p.add_argument("--out-net-name", required=True)
    p.add_argument("--dataset-name", default="Dataset509_ARCADE_1x1_4c")
    p.add_argument("--layer-bits-file", required=True, type=Path)
    p.add_argument("--plans-name", default="nnUNetPlans")
    p.add_argument("--configuration", default="2d")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-calibration-images", type=int, default=64)
    p.add_argument("--calibration-seed", type=int, default=0)
    p.add_argument("--pad-to-multiple", type=int, default=0)
    p.add_argument("--no-decoder-prelu", action="store_true", help="source is encoder-only PReLU: decoder sites stay QuantReLU")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    with open(args.layer_bits_file) as f:
        layer_bits = json.load(f)
    source_folder = _relu.NNUNET_RESULTS / args.dataset_name / f"{args.source_net_name}__{args.plans_name}__{args.configuration}"
    source_ckpt = source_folder / f"fold_{args.fold}" / args.source_checkpoint_name
    if not source_ckpt.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_ckpt}")

    print(f"Loading source FP32 checkpoint: {source_ckpt}")
    model = LayerQuantEnetFINN.from_pretrained(
        source_ckpt, layer_bits["layer_weight_bits"], layer_bits["layer_act_bits"],
        in_channels=1, out_channels=5, channels=_relu.CHANNELS, bottlenecks_per_stage=_relu.BOTTLENECKS_PER_STAGE,
        context_pattern=_relu.CONTEXT_PATTERN, decoder_type=_relu.DECODER_TYPE,
        prelu=True, decoder_prelu=not args.no_decoder_prelu,
    )

    print(f"Calibrating on real preprocessed images (device={args.device})...")
    batches = _relu.load_calibration_batches(args.dataset_name, args.n_calibration_images, seed=args.calibration_seed, pad_multiple=args.pad_to_multiple)
    n_used = _relu.calibrate(model, batches, args.device, seed=args.calibration_seed, train_mode=False)
    print(f"Calibration used {n_used}/{len(batches)} images.")
    model.to("cpu")
    print(f"Clamped {_relu.clamp_quantizer_scales(model)} quantizer scale(s) below the fp16-safe floor.")

    out_folder = _relu.NNUNET_RESULTS / args.dataset_name / f"{args.out_net_name}__{args.plans_name}__{args.configuration}"
    out_fold = out_folder / f"fold_{args.fold}"
    out_fold.mkdir(parents=True, exist_ok=True)
    for meta in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
        if (source_folder / meta).exists():
            (out_folder / meta).write_bytes((source_folder / meta).read_bytes())

    checkpoint = dict(torch.load(source_ckpt, map_location="cpu", weights_only=False))
    weights = dict(model.state_dict())
    weights.update(dict(model.named_parameters(remove_duplicate=False)))
    checkpoint["network_weights"] = weights
    checkpoint["trainer_name"] = TRAINER_NAME
    torch.save(checkpoint, out_fold / "checkpoint_best.pth")
    print(f"Saved calibrated checkpoint: {out_fold / 'checkpoint_best.pth'}")


if __name__ == "__main__":
    main()
