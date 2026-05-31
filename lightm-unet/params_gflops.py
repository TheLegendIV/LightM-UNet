import os
import json
from pathlib import Path

import torch
from thop import profile


def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    dataset_name = "Dataset501_ARCADE"
    configuration = "2d"
    fold = 0

    nnunet_preprocessed = os.environ.get("nnUNet_preprocessed")

    if nnunet_preprocessed is None:
        raise RuntimeError("Environment variable nnUNet_preprocessed is not set.")

    plans_path = Path(nnunet_preprocessed) / dataset_name / "nnUNetPlans.json"
    dataset_json_path = Path(nnunet_preprocessed) / dataset_name / "dataset.json"

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

    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLightMUNet import (
        nnUNetTrainerLightMUNet,
    )

    # -------------------------------------------------------------------------
    # Important:
    # LightM-UNet uses mamba_ssm CUDA kernels. Parameter counting can be done
    # without a forward pass, but GFLOPs profiling requires running a forward pass.
    # Therefore, this script must run inside a GPU job.
    # -------------------------------------------------------------------------
    print("\n=== CUDA check ===")
    print("torch:", torch.__version__)
    print("torch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this script inside a GPU job. "
            "LightM-UNet uses mamba_ssm CUDA kernels, so THOP/GFLOPs profiling "
            "cannot run on CPU."
        )

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    trainer = nnUNetTrainerLightMUNet(
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

    total_params, trainable_params = count_parameters(model)

    print("\n=== Parameters ===")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters:     {total_params / 1e6:.3f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.3f} M")

    # ARCADE is grayscale, so input channels = 1
    h, w = patch_size
    dummy_input = torch.randn(1, 1, h, w, device=device)

    print("\n=== FLOPs estimate ===")
    print(f"Input shape: {tuple(dummy_input.shape)}")

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