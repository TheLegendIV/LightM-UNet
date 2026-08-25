"""Trainer stub for nnUNetTrainerENet_5_6_separable_dense_dilation's ("S5.6")
PER-BLOCK-quantized mirror (nnunetv2.nets.QuantENet5_6Block.QuantENet5_6Block)
-- see nnUNetTrainerENetQuant26_5_w24Block.py's own module docstring for the
general shape of why this class exists (checkpoint-driven architecture
reconstruction for nnUNetv2_predict).

ENET5_6_BLOCK_BITS_FILE (required): path to compression/hawq/ilp_search.py's
per-block output JSON ({"stage_weight_bits": {...}, "stage_act_bits": {...}},
one entry per QuantENet5_6Block.BLOCK_NAMES) -- e.g.
compression/hawq/block_bits_5_6.json.

ENET5_6_LEAKY_SLOPE_MAP_FILE (optional but expected for the real per-block
QAT run): path to compression/post-quantization/extract_leaky_slope_map.py's
output JSON. S5.6 is prelu_variant="standard" (real per-channel PReLU, NOT a
trained nonneg_block scalar) -- see QuantENet5_6Block.py's own module
docstring for why this file's own default source (--prelu-variant standard's
post-hoc per-channel average) is NOT a like-for-like substitute for a real
nonneg_block-trained scalar, and why QAT (real backprop over many epochs,
letting every weight re-adapt to the now-fixed per-block slope) is exactly
the setting where that flagged choice is actually defensible, unlike a bare
PTQ calibration.

ENET5_6_PRETRAINED_CHECKPOINT (optional): path to a real FP32
nnUNetTrainerENet_5_6_separable_dense_dilation checkpoint to warm-start from
(conv/BN weights transferred by name+shape via QuantENet5_6Block.
from_pretrained's own strict=False logic) -- omit to cold-start instead.
"""
import json
import os

import torch
from torch import nn

from nnunetv2.nets.QuantENet5_6Block import QuantENet5_6Block, BLOCK_NAMES
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


class nnUNetTrainerENetQuant5_6Block(nnUNetTrainerENet):
    def __init__(
        self, plans: dict, configuration: str, fold: int, dataset_json: dict,
        unpack_dataset: bool = True, device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        block_bits_file = os.environ.get("ENET5_6_BLOCK_BITS_FILE")
        if not block_bits_file:
            raise ValueError(
                "ENET5_6_BLOCK_BITS_FILE must be set -- path to compression/hawq/ilp_search.py's "
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
            raise ValueError("QuantENet5_6Block is a 2D architecture. Use the nnU-Net 2d configuration.")
        label_manager = plans_manager.get_label_manager(dataset_json)
        if num_input_channels != 1 or label_manager.num_segmentation_heads != 5:
            raise ValueError(
                f"QuantENet5_6Block is hardcoded to in_channels=1, out_channels=5 (Dataset509_ARCADE_1x1_4c) -- "
                f"got num_input_channels={num_input_channels}, num_segmentation_heads={label_manager.num_segmentation_heads}. "
                f"Wrong dataset/plans for this trainer."
            )

        block_bits_file = os.environ.get("ENET5_6_BLOCK_BITS_FILE")
        block_bits = _load_block_bits(block_bits_file)

        slope_map_file = os.environ.get("ENET5_6_LEAKY_SLOPE_MAP_FILE")
        leaky_slope_map = None
        if slope_map_file:
            with open(slope_map_file) as f:
                leaky_slope_map = json.load(f)

        pretrained_checkpoint = os.environ.get("ENET5_6_PRETRAINED_CHECKPOINT")
        if pretrained_checkpoint:
            return QuantENet5_6Block.from_pretrained(
                pretrained_checkpoint, block_bits["stage_weight_bits"], block_bits["stage_act_bits"], leaky_slope_map,
            )
        return QuantENet5_6Block(block_bits["stage_weight_bits"], block_bits["stage_act_bits"], leaky_slope_map)
