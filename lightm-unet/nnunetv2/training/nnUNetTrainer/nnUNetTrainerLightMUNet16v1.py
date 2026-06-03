from torch import nn

from nnunetv2.nets.LightMUNet import LightMUNet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLightMUNet import nnUNetTrainerLightMUNet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerLightMUNet16v1(nnUNetTrainerLightMUNet):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)

        return LightMUNet(
            spatial_dims=len(configuration_manager.patch_size),
            init_filters=16,
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            blocks_down=[2, 3, 3, 4],
            blocks_up=[1, 1, 1],
        )
