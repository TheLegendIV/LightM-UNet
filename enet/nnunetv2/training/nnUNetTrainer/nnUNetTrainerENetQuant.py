import os

import torch
from torch import nn

from nnunetv2.nets.QuantENet import QuantENet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import (
    _parse_bool_env,
    _parse_bottlenecks,
    _parse_channels,
    nnUNetTrainerENet,
)
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerENetQuant(nnUNetTrainerENet):
    """QAT counterpart of nnUNetTrainerENet -- same env-var-driven config
    (ENET_CHANNELS/ENET_BOTTLENECKS/ENET_DECODER_TYPE/ENET_USE_*, same
    optimizer/LR/epoch overrides, all inherited unchanged), plus
    ENET_QUANT_BITS for homogeneous weight+activation bit-width
    (agent_instructions_1.yaml's early_probes.p2_quantization_probe /
    stage_4_quantization.4_1_homogeneous). Heterogeneous per-layer
    quantization (4_2) is not built here -- QuantENet only supports one
    bit-width network-wide so far.

    Only builds QuantENet in place of ENet; no FP32 reference point here --
    ENET_QUANT_BITS=32 isn't a meaningful QAT config (Brevitas quant layers
    at bit_width=32 aren't "no quantization", just wasteful int32 fake-quant).
    The FP32 baseline for comparison is whatever Stage 1/Stage 2 already
    trained with plain nnUNetTrainerENet on the same channels/bottlenecks.
    """

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.quant_bits = int(os.environ.get("ENET_QUANT_BITS", "8"))
        if self.quant_bits >= 32:
            raise ValueError(
                "ENET_QUANT_BITS must be < 32 (2/4/8 for the homogeneous sweep) -- "
                "nnUNetTrainerENetQuant is for QAT runs; use plain nnUNetTrainerENet "
                "for the FP32 reference point."
            )

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Brevitas's parameter-scaling modules (e.g. StatsFromParameterScaling)
        override nn.Module._load_from_state_dict and re-materialize their
        `scaling_impl.value` parameter from scratch while loading, which drops
        it back onto CPU regardless of the checkpoint tensor's own device --
        confirmed directly (548 stray CPU params after a plain load_state_dict
        on a freshly-built, already-.to(cuda)'d QuantENet, 0 before) while
        resuming 3a_quant_O4_int8_no_asym via --c. A fresh (non-resumed) run
        never calls load_state_dict on this model at all, so it never hits
        this path -- only continuation runs need this re-placement.
        """
        super().load_checkpoint(filename_or_checkpoint)
        self.network = self.network.to(self.device)

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError("ENet is a 2D architecture. Use the nnU-Net 2d configuration.")
        label_manager = plans_manager.get_label_manager(dataset_json)
        channels = _parse_channels(os.environ.get("ENET_CHANNELS", "20,72,144,72,20"))
        bottlenecks_per_stage = _parse_bottlenecks(os.environ.get("ENET_BOTTLENECKS", "4,8,8,2,1"))
        decoder_type = os.environ.get("ENET_DECODER_TYPE", "max_unpool")
        quant_bits = int(os.environ.get("ENET_QUANT_BITS", "8"))
        return QuantENet(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            channels=channels,
            bottlenecks_per_stage=bottlenecks_per_stage,
            decoder_type=decoder_type,
            use_dilated=_parse_bool_env("ENET_USE_DILATED", True),
            use_asymmetric=_parse_bool_env("ENET_USE_ASYMMETRIC", True),
            use_strided=_parse_bool_env("ENET_USE_STRIDED", True),
            use_dsc=_parse_bool_env("ENET_USE_DSC", False),
            weight_bit_width=quant_bits,
            act_bit_width=quant_bits,
        )
