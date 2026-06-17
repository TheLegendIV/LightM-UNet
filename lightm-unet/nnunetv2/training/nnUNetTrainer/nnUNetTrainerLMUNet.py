import os
import random
from datetime import datetime

import numpy as np
from torch import nn
import torch
from torch.optim import AdamW

from nnunetv2.nets.LMUNet import LMUNet
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLightMUNet import nnUNetTrainerLightMUNet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


def _parse_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(channels) != 6:
        raise ValueError("LMUNET_CHANNELS must contain six comma-separated integers.")
    return channels


class nnUNetTrainerLMUNet(nnUNetTrainerLightMUNet):
    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda')
        ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        if os.environ.get("LMUNET_SEED"):
            seed = int(os.environ["LMUNET_SEED"])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.initial_lr = 1e-3
        self.weight_decay = 1e-2
        if os.environ.get("LMUNET_OUTPUT_FOLDER"):
            self.output_folder = os.environ["LMUNET_OUTPUT_FOLDER"]
            self.output_folder_base = os.path.dirname(self.output_folder)
            os.makedirs(self.output_folder, exist_ok=True)
            timestamp = datetime.now()
            self.log_file = os.path.join(
                self.output_folder,
                "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" % (
                    timestamp.year,
                    timestamp.month,
                    timestamp.day,
                    timestamp.hour,
                    timestamp.minute,
                    timestamp.second,
                ),
            )
        if os.environ.get("LMUNET_EPOCHS"):
            self.num_epochs = int(os.environ["LMUNET_EPOCHS"])
        if os.environ.get("LMUNET_BATCH_SIZE"):
            self.batch_size = int(os.environ["LMUNET_BATCH_SIZE"])
        if os.environ.get("LMUNET_ITERATIONS_PER_EPOCH"):
            self.num_iterations_per_epoch = int(os.environ["LMUNET_ITERATIONS_PER_EPOCH"])
        if os.environ.get("LMUNET_VAL_ITERATIONS_PER_EPOCH"):
            self.num_val_iterations_per_epoch = int(os.environ["LMUNET_VAL_ITERATIONS_PER_EPOCH"])
        if os.environ.get("LMUNET_DISABLE_CHECKPOINTING", "0") == "1":
            self.disable_checkpointing = True
            self.save_every = 10**9

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        channels = _parse_channels(os.environ.get("LMUNET_CHANNELS", "12,24,44,44,60,60"))
        edge_channels = int(os.environ.get("LMUNET_EDGE_CHANNELS", "20"))

        return LMUNet(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            channels=channels,
            edge_channels=edge_channels,
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
        if os.environ.get("LMUNET_SKIP_FINAL_VALIDATION", "0") == "1":
            self.print_to_log_file("Skipping final full validation because LMUNET_SKIP_FINAL_VALIDATION=1")
            return
        return super().perform_actual_validation(save_probabilities)

    def plot_network_architecture(self):
        if os.environ.get("LMUNET_SKIP_ARCH_PLOT", "0") == "1":
            self.print_to_log_file("Skipping network architecture plot because LMUNET_SKIP_ARCH_PLOT=1")
            return
        return super().plot_network_architecture()
