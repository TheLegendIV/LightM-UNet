import os

import torch
from torch import nn
from torch.optim import AdamW

from nnunetv2.nets.ENetUpscaled import ENetUpscaled
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLightMUNet import nnUNetTrainerLightMUNet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


def _parse_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(channels) != 5:
        raise ValueError(
            "ENETUSC_CHANNELS must contain five comma-separated integers: "
            "ch0,ch1,ch2,ch3,ch4"
        )
    return channels


class nnUNetTrainerENetUpscaled(nnUNetTrainerLightMUNet):
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
        self.initial_lr = float(os.environ.get("ENETUSC_LR", "1e-3"))
        self.weight_decay = float(os.environ.get("ENETUSC_WEIGHT_DECAY", "1e-2"))
        if os.environ.get("ENETUSC_EPOCHS"):
            self.num_epochs = int(os.environ["ENETUSC_EPOCHS"])
        if os.environ.get("ENETUSC_BATCH_SIZE"):
            self.batch_size = int(os.environ["ENETUSC_BATCH_SIZE"])
        if os.environ.get("ENETUSC_ITERATIONS_PER_EPOCH"):
            self.num_iterations_per_epoch = int(os.environ["ENETUSC_ITERATIONS_PER_EPOCH"])
        if os.environ.get("ENETUSC_VAL_ITERATIONS_PER_EPOCH"):
            self.num_val_iterations_per_epoch = int(os.environ["ENETUSC_VAL_ITERATIONS_PER_EPOCH"])
        if os.environ.get("ENETUSC_DISABLE_CHECKPOINTING", "0") == "1":
            self.disable_checkpointing = True
            self.save_every = 10**9
        if os.environ.get("ENETUSC_OUTPUT_FOLDER"):
            self.output_folder = os.environ["ENETUSC_OUTPUT_FOLDER"]
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
            raise ValueError("ENetUpscaled is a 2D architecture. Use the nnU-Net 2d configuration.")
        label_manager = plans_manager.get_label_manager(dataset_json)
        channels = _parse_channels(
            os.environ.get("ENETUSC_CHANNELS", "20,72,72,144,256")
        )
        asym_heavy_stage3 = os.environ.get("ENETUSC_ASYM_STAGE3", "0") == "1"
        return ENetUpscaled(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            channels=channels,
            asym_heavy_stage3=asym_heavy_stage3,
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
        if os.environ.get("ENETUSC_SKIP_FINAL_VALIDATION", "0") == "1":
            self.print_to_log_file("Skipping final full validation because ENETUSC_SKIP_FINAL_VALIDATION=1")
            return
        return super().perform_actual_validation(save_probabilities)

    def plot_network_architecture(self):
        if os.environ.get("ENETUSC_SKIP_ARCH_PLOT", "0") == "1":
            self.print_to_log_file("Skipping network architecture plot because ENETUSC_SKIP_ARCH_PLOT=1")
            return
        return super().plot_network_architecture()
