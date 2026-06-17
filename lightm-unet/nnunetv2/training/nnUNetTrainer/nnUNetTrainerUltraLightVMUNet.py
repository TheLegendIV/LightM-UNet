import os
import random
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from nnunetv2.nets.UVMUNet import UVMUNet
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLightMUNet import nnUNetTrainerLightMUNet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


def _parse_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(channels) != 6:
        raise ValueError("ULVMUNET_CHANNELS must contain six comma-separated integers.")
    return channels


class nnUNetTrainerUltraLightVMUNet(nnUNetTrainerLightMUNet):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        if os.environ.get("ULVMUNET_SEED"):
            seed = int(os.environ["ULVMUNET_SEED"])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.initial_lr = float(os.environ.get("ULVMUNET_LR", "1e-3"))
        self.weight_decay = float(os.environ.get("ULVMUNET_WEIGHT_DECAY", "1e-2"))
        if os.environ.get("ULVMUNET_OUTPUT_FOLDER"):
            self.output_folder = os.environ["ULVMUNET_OUTPUT_FOLDER"]
            self.output_folder_base = os.path.dirname(self.output_folder)
            os.makedirs(self.output_folder, exist_ok=True)
            timestamp = datetime.now()
            self.log_file = os.path.join(
                self.output_folder,
                "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt"
                % (
                    timestamp.year,
                    timestamp.month,
                    timestamp.day,
                    timestamp.hour,
                    timestamp.minute,
                    timestamp.second,
                ),
            )
        if os.environ.get("ULVMUNET_EPOCHS"):
            self.num_epochs = int(os.environ["ULVMUNET_EPOCHS"])
        if os.environ.get("ULVMUNET_BATCH_SIZE"):
            self.batch_size = int(os.environ["ULVMUNET_BATCH_SIZE"])
        if os.environ.get("ULVMUNET_ITERATIONS_PER_EPOCH"):
            self.num_iterations_per_epoch = int(os.environ["ULVMUNET_ITERATIONS_PER_EPOCH"])
        if os.environ.get("ULVMUNET_VAL_ITERATIONS_PER_EPOCH"):
            self.num_val_iterations_per_epoch = int(os.environ["ULVMUNET_VAL_ITERATIONS_PER_EPOCH"])
        if os.environ.get("ULVMUNET_DISABLE_CHECKPOINTING", "0") == "1":
            self.disable_checkpointing = True
            self.save_every = 10**9

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError("UltraLightVMUNet is a 2D architecture. Use the nnU-Net 2d configuration.")
        label_manager = plans_manager.get_label_manager(dataset_json)
        channels = _parse_channels(os.environ.get("ULVMUNET_CHANNELS", "8,16,24,32,48,64"))
        bridge = os.environ.get("ULVMUNET_BRIDGE", "1") != "0"
        split_att = os.environ.get("ULVMUNET_SPLIT_ATT", "fc")

        return UVMUNet(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            channels=channels,
            split_att=split_att,
            bridge=bridge,
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
        if os.environ.get("ULVMUNET_SKIP_FINAL_VALIDATION", "0") == "1":
            self.print_to_log_file("Skipping final full validation because ULVMUNET_SKIP_FINAL_VALIDATION=1")
            return
        return super().perform_actual_validation(save_probabilities)

    def plot_network_architecture(self):
        if os.environ.get("ULVMUNET_SKIP_ARCH_PLOT", "0") == "1":
            self.print_to_log_file("Skipping network architecture plot because ULVMUNET_SKIP_ARCH_PLOT=1")
            return
        return super().plot_network_architecture()
