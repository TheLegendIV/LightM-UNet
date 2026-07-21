import importlib
import json
import os
from pathlib import Path

import torch


def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_trainer_class(trainer_name: str):
    module = importlib.import_module(f"nnunetv2.training.nnUNetTrainer.{trainer_name}")
    return getattr(module, trainer_name), module


def main():
    dataset_name = os.environ.get("LMUNET_STATS_DATASET", "Dataset501_ARCADE")
    configuration = os.environ.get("LMUNET_STATS_CONFIGURATION", "2d")
    fold = int(os.environ.get("LMUNET_STATS_FOLD", "0"))
    trainer_name = os.environ.get("LMUNET_STATS_TRAINER", "nnUNetTrainerLMUNet")

    nnunet_preprocessed = os.environ.get("nnUNet_preprocessed")

    if nnunet_preprocessed is None:
        raise RuntimeError("Environment variable nnUNet_preprocessed is not set.")

    plans_path = Path(nnunet_preprocessed) / dataset_name / "nnUNetPlans.json"
    dataset_json_path = Path(nnunet_preprocessed) / dataset_name / "dataset.json"

    print("Using trainer:", trainer_name)
    print("Using plans:", plans_path)
    print("Using dataset.json:", dataset_json_path)

    with open(plans_path, "r") as f:
        plans = json.load(f)

    with open(dataset_json_path, "r") as f:
        dataset_json = json.load(f)

    cfg = plans["configurations"][configuration]
    patch_size = cfg["patch_size"]

    print("\n=== Plan info ===")
    print("Configuration:", configuration)
    print("Patch size:", patch_size)
    print("Batch size:", cfg["batch_size"])

    trainer_class, trainer_module = load_trainer_class(trainer_name)

    print("\n=== CUDA check ===")
    print("torch:", torch.__version__)
    print("torch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this script inside a GPU job. "
            "Mamba CUDA kernels require GPU execution for profiling."
        )

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    trainer = trainer_class(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        unpack_dataset=False,
        device=device,
    )

    trainer.initialize()
    model = trainer.network
    model.eval()
    model.to(device)

    print("\n=== Model source ===")
    print("Trainer module:", trainer_module.__file__)
    print("Model class:", model.__class__.__name__)
    print("Model channels:", getattr(model, "channels", "N/A"))
    print("Model in_channels:", getattr(model, "in_channels", "N/A"))
    print("Model out_channels:", getattr(model, "out_channels", "N/A"))

    total_params, trainable_params = count_parameters(model)

    print("\n=== Parameters ===")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters:     {total_params / 1e6:.3f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.3f} M")

    h, w = patch_size
    in_channels = getattr(model, "in_channels", 1)
    dummy_input = torch.randn(1, in_channels, h, w, device=device)

    print("\n=== FLOPs estimate ===")
    print(f"Input shape: {tuple(dummy_input.shape)}")

    try:
        from thop import profile
    except ImportError:
        print("THOP is not installed, so FLOPs were not estimated.")
        print("Install it with: python -m pip install thop")
        return

    torch.cuda.empty_cache()

    with torch.no_grad():
        macs, thop_params = profile(model, inputs=(dummy_input,), verbose=False)

    gmacs = macs / 1e9
    gflops = 2 * gmacs

    print(f"MACs:   {gmacs:.3f} GMACs")
    print(f"FLOPs:  {gflops:.3f} GFLOPs")
    print(f"THOP parameters: {thop_params / 1e6:.3f} M")

    print("\nNote:")
    print("GFLOPs are estimated with THOP.")
    print("FLOPs are reported using the convention FLOPs = 2 x MACs.")
    print("Custom Mamba/selective-scan CUDA operations may be undercounted.")
    print("Parameter count is reliable; GFLOPs should be treated as approximate.")


if __name__ == "__main__":
    main()
