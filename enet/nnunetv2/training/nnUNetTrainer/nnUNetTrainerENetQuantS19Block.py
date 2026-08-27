"""Trainer stub for nnUNetTrainerENet_19_reginterleaved_separable_nonneg_block_double_mid's
("S19") PER-BLOCK-quantized mirror (nnunetv2.nets.QuantENetS19Block.QuantENetS19Block)
-- see nnUNetTrainerENetQuant26_5_w24Block.py's own module docstring for why
this class exists (checkpoint-driven architecture reconstruction for
nnUNetv2_predict, not a real training loop -- compression/post-quantization/
ptq_block.py never instantiates or trains through this class).

ENET_S19_BLOCK_BITS_FILE (required): path to compression/hawq/
ilp_search.py's per-block output JSON, e.g.
compression/hawq/block_bits_s19.json.

ENET_S19_LEAKY_SLOPE_MAP_FILE (optional but not really -- see
nnUNetTrainerENetQuant23_1.py's own note, identical reasoning): S19 is
prelu_variant="nonneg_block", a REAL trained per-block scalar (unlike
26_5_w24's post-hoc average) -- compression/post-quantization/slope_maps/
19_reginterleaved_separable_nonneg_block_double_mid.json. Skipping this
silently builds plain QuantReLU everywhere, a genuine architecture
mismatch (see ptq_s19_double_mid_int8.job's own documented incident:
dice=0.4383 with no slope map vs. the real number with one).

ENET_S19_PRETRAINED_CHECKPOINT (optional but effectively the real default
-- see hawq_23_1_qat_stage_bits.job's own precedent of warm-starting from
the architecture's own FP32 lineage): path to a real FP32
nnUNetTrainerENet_19_reginterleaved_separable_nonneg_block_double_mid
checkpoint to warm-start from. Omit to cold-start instead.

ENET_S19_TRAINABLE_SLOPE (optional, default "1" i.e. True): set to "0" to
build with the ORIGINAL frozen-slope QuantDecomposedLeakyAct behavior
instead of QuantENetS19Block.py's current trainable_slope=True default --
exists purely so a controlled ablation can cross-compare a trainable-slope
run against a frozen-slope run on the EXACT same bit assignment/warm-start/
LR schedule, isolating the slope mechanism from every other variable. Every
real (non-diagnostic) job should leave this unset.
"""
import json
import os

import torch
from torch import nn

from nnunetv2.nets.QuantENetS19Block import QuantENetS19Block, BLOCK_NAMES
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import nnUNetTrainerENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


def _load_block_bits(path: str) -> dict:
    with open(path) as f:
        block_bits = json.load(f)
    for key in ("stage_weight_bits", "stage_act_bits"):
        missing = [b for b in BLOCK_NAMES if b not in block_bits.get(key, {})]
        if missing:
            raise ValueError(f"{path}'s {key!r} is missing entries for blocks: {missing}")
    return block_bits


class nnUNetTrainerENetQuantS19Block(nnUNetTrainerENet):
    def __init__(
        self, plans: dict, configuration: str, fold: int, dataset_json: dict,
        unpack_dataset: bool = True, device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        block_bits_file = os.environ.get("ENET_S19_BLOCK_BITS_FILE")
        if not block_bits_file:
            raise ValueError(
                "ENET_S19_BLOCK_BITS_FILE must be set -- path to compression/hawq/ilp_search.py's "
                "per-block output JSON ({'stage_weight_bits': {...}, 'stage_act_bits': {...}})."
            )
        self._block_bits = _load_block_bits(block_bits_file)

    def load_checkpoint(self, filename_or_checkpoint) -> None:
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
            raise ValueError("QuantENetS19Block is a 2D architecture. Use the nnU-Net 2d configuration.")
        label_manager = plans_manager.get_label_manager(dataset_json)
        if num_input_channels != 1 or label_manager.num_segmentation_heads != 5:
            raise ValueError(
                f"QuantENetS19Block is hardcoded to in_channels=1, out_channels=5 (Dataset509_ARCADE_1x1_4c) -- "
                f"got num_input_channels={num_input_channels}, num_segmentation_heads={label_manager.num_segmentation_heads}. "
                f"Wrong dataset/plans for this trainer."
            )

        block_bits_file = os.environ.get("ENET_S19_BLOCK_BITS_FILE")
        block_bits = _load_block_bits(block_bits_file)

        slope_map_file = os.environ.get("ENET_S19_LEAKY_SLOPE_MAP_FILE")
        leaky_slope_map = None
        if slope_map_file:
            with open(slope_map_file) as f:
                leaky_slope_map = json.load(f)

        trainable_slope = os.environ.get("ENET_S19_TRAINABLE_SLOPE", "1") != "0"

        pretrained_checkpoint = os.environ.get("ENET_S19_PRETRAINED_CHECKPOINT")
        if pretrained_checkpoint:
            return QuantENetS19Block.from_pretrained(
                pretrained_checkpoint, block_bits["stage_weight_bits"], block_bits["stage_act_bits"], leaky_slope_map,
                trainable_slope=trainable_slope,
            )
        return QuantENetS19Block(
            block_bits["stage_weight_bits"], block_bits["stage_act_bits"], leaky_slope_map,
            trainable_slope=trainable_slope,
        )
