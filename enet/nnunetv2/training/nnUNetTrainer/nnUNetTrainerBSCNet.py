import os
import random

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from nnunetv2.nets.BSCNet import BSCNet
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLightMUNet import nnUNetTrainerLightMUNet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerBSCNet(nnUNetTrainerLightMUNet):
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
        if os.environ.get("BSCNET_SEED"):
            seed = int(os.environ["BSCNET_SEED"])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.initial_lr = float(os.environ.get("BSCNET_LR", "1e-3"))
        self.weight_decay = float(os.environ.get("BSCNET_WEIGHT_DECAY", "1e-2"))
        if os.environ.get("BSCNET_EPOCHS"):
            self.num_epochs = int(os.environ["BSCNET_EPOCHS"])
        if os.environ.get("BSCNET_BATCH_SIZE"):
            self.batch_size = int(os.environ["BSCNET_BATCH_SIZE"])
        if os.environ.get("BSCNET_ITERATIONS_PER_EPOCH"):
            self.num_iterations_per_epoch = int(os.environ["BSCNET_ITERATIONS_PER_EPOCH"])
        if os.environ.get("BSCNET_VAL_ITERATIONS_PER_EPOCH"):
            self.num_val_iterations_per_epoch = int(os.environ["BSCNET_VAL_ITERATIONS_PER_EPOCH"])
        if os.environ.get("BSCNET_DISABLE_CHECKPOINTING", "0") == "1":
            self.disable_checkpointing = True
            self.save_every = 10**9
        if os.environ.get("BSCNET_OUTPUT_FOLDER"):
            self.output_folder = os.environ["BSCNET_OUTPUT_FOLDER"]
            self.output_folder_base = os.path.dirname(self.output_folder)
            os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError("BSCNet is a 2D architecture. Use the nnU-Net 2d configuration.")
        label_manager = plans_manager.get_label_manager(dataset_json)
        planes = int(os.environ.get("BSCNET_PLANES", "32"))
        head_planes = int(os.environ.get("BSCNET_HEAD_PLANES", "64"))
        return BSCNet(
            in_channels=num_input_channels,
            num_classes=label_manager.num_segmentation_heads,
            planes=planes,
            head_planes=head_planes,
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
        if os.environ.get("BSCNET_SKIP_FINAL_VALIDATION", "0") == "1":
            self.print_to_log_file("Skipping final full validation because BSCNET_SKIP_FINAL_VALIDATION=1")
            return
        return super().perform_actual_validation(save_probabilities)

    def plot_network_architecture(self):
        if os.environ.get("BSCNET_SKIP_ARCH_PLOT", "0") == "1":
            self.print_to_log_file("Skipping network architecture plot because BSCNET_SKIP_ARCH_PLOT=1")
            return
        return super().plot_network_architecture()
