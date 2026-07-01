import os

import torch
from torch import nn
from torch.optim import AdamW

from nnunetv2.nets.ENetGlobalCtx import VALID_EXPERIMENTS, build_gctx_model
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLightMUNet import nnUNetTrainerLightMUNet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


def _parse_channels(value: str) -> tuple[int, ...]:
    parts = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    if len(parts) != 5:
        raise ValueError("GCX_CHANNELS must be five comma-separated ints: ch0,ch1,ch23,ch4,ch5")
    return parts


class nnUNetTrainerENetGlobalCtx(nnUNetTrainerLightMUNet):

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = float(os.environ.get("GCX_LR", "1e-3"))
        self.weight_decay = float(os.environ.get("GCX_WEIGHT_DECAY", "1e-2"))
        if os.environ.get("GCX_EPOCHS"):
            self.num_epochs = int(os.environ["GCX_EPOCHS"])
        if os.environ.get("GCX_BATCH_SIZE"):
            self.batch_size = int(os.environ["GCX_BATCH_SIZE"])
        if os.environ.get("GCX_ITERATIONS_PER_EPOCH"):
            self.num_iterations_per_epoch = int(os.environ["GCX_ITERATIONS_PER_EPOCH"])
        if os.environ.get("GCX_VAL_ITERATIONS_PER_EPOCH"):
            self.num_val_iterations_per_epoch = int(os.environ["GCX_VAL_ITERATIONS_PER_EPOCH"])
        if os.environ.get("GCX_DISABLE_CHECKPOINTING", "0") == "1":
            self.disable_checkpointing = True
            self.save_every = 10**9
        if os.environ.get("GCX_OUTPUT_FOLDER"):
            self.output_folder = os.environ["GCX_OUTPUT_FOLDER"]
            self.output_folder_base = os.path.dirname(self.output_folder)
            os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError("ENetGlobalCtx is a 2D architecture. Use nnU-Net 2d configuration.")
        label_manager = plans_manager.get_label_manager(dataset_json)
        experiment = os.environ.get("GCX_EXPERIMENT", "G1")
        if experiment not in VALID_EXPERIMENTS:
            raise ValueError(
                f"GCX_EXPERIMENT='{experiment}' is not valid. Valid: {sorted(VALID_EXPERIMENTS)}"
            )
        channels = _parse_channels(os.environ.get("GCX_CHANNELS", "20,72,144,72,20"))
        return build_gctx_model(
            experiment=experiment,
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            channels=channels,
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=0.9)
        return optimizer, scheduler

    def perform_actual_validation(self, save_probabilities: bool = False):
        if os.environ.get("GCX_SKIP_FINAL_VALIDATION", "0") == "1":
            self.print_to_log_file("Skipping final validation (GCX_SKIP_FINAL_VALIDATION=1)")
            return
        return super().perform_actual_validation(save_probabilities)

    def plot_network_architecture(self):
        if os.environ.get("GCX_SKIP_ARCH_PLOT", "0") == "1":
            self.print_to_log_file("Skipping architecture plot (GCX_SKIP_ARCH_PLOT=1)")
            return
        return super().plot_network_architecture()
