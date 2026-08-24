"""Trainer stub for nnUNetTrainerENet_26_5_w24's PER-BLOCK-quantized mirror
(nnunetv2.nets.QuantENet26_5_w24.QuantENet26_5_w24) -- exists so a saved PTQ
checkpoint's trainer_name resolves to a real class nnUNetv2_predict's
recursive_find_python_class can locate, whose build_network_architecture
staticmethod reconstructs the exact per-block-quantized architecture
(compression/post-quantization/ptq_block.py never runs this class's
__init__ or any real nnU-Net training loop -- it builds QuantENet26_5_w24
directly, calibrates, and stamps this class's NAME into the saved
checkpoint purely so nnUNetv2_predict_from_modelfolder's checkpoint-driven
class lookup, called via compression/collect_results.py, can rebuild the
right network shape before loading the calibrated state_dict).

A real QAT run (not just PTQ) WOULD use this class's own __init__/
load_checkpoint the normal nnUNetTrainerENet way -- kept structurally
parallel to nnUNetTrainerENetQuant23_1.py for that reason, even though
today's only caller is ptq_block.py.

ENET26_5_W24_BLOCK_BITS_FILE (required): path to compression/hawq/
ilp_search.py's per-block output JSON ({"stage_weight_bits": {...},
"stage_act_bits": {...}}, one entry per QuantENet26_5_w24.BLOCK_NAMES) --
e.g. compression/hawq/block_bits_26_5_w24.json.

ENET26_5_W24_LEAKY_SLOPE_MAP_FILE (optional): path to compression/post-
quantization/extract_leaky_slope_map.py's output JSON. See QuantENet26_5_w24.py's
own module docstring for why this file's own default source (--prelu-variant
standard's post-hoc per-channel average) is NOT a like-for-like substitute
for a real nonneg_block-trained scalar -- passing it here is a deliberate,
flagged choice (decomposed-nonneg-PReLU QAT), not the "safe default" it
would be for an actually nonneg_block-trained checkpoint like S19's. QAT
(real backprop over many epochs) is exactly the setting where that
flagged choice is actually defensible -- unlike a bare PTQ calibration,
every weight here gets to re-adapt to the now-fixed per-block slope, the
same "let training absorb the mismatch" logic nnUNetTrainerENetQuant23_1.py's
own module docstring already relies on for its own from_pretrained warm
start.

ENET26_5_W24_PRETRAINED_CHECKPOINT (optional): path to a real FP32
nnUNetTrainerENet_26_5_w24 checkpoint to warm-start from (conv/BN weights
transferred by name+shape via QuantENet26_5_w24.from_pretrained's own
strict=False logic) -- omit to cold-start instead. See
nnUNetTrainerENetQuant23_1.py's own note on why warm-starting from the
architecture's own FP32 lineage is the default choice elsewhere in this
repo's HAWQ QAT jobs.
"""
import json
import os

import torch
from torch import nn

from nnunetv2.nets.QuantENet26_5_w24 import QuantENet26_5_w24, BLOCK_NAMES
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


class nnUNetTrainerENetQuant26_5_w24Block(nnUNetTrainerENet):
    def __init__(
        self, plans: dict, configuration: str, fold: int, dataset_json: dict,
        unpack_dataset: bool = True, device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        block_bits_file = os.environ.get("ENET26_5_W24_BLOCK_BITS_FILE")
        if not block_bits_file:
            raise ValueError(
                "ENET26_5_W24_BLOCK_BITS_FILE must be set -- path to compression/hawq/ilp_search.py's "
                "per-block output JSON ({'stage_weight_bits': {...}, 'stage_act_bits': {...}})."
            )
        self._block_bits = _load_block_bits(block_bits_file)

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Same Brevitas parameter-scaling re-materialization quirk
        nnUNetTrainerENetQuant.py/nnUNetTrainerENetQuant23_1.py already
        document -- only bites a resumed (--c) run."""
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
            raise ValueError("QuantENet26_5_w24 is a 2D architecture. Use the nnU-Net 2d configuration.")
        label_manager = plans_manager.get_label_manager(dataset_json)
        if num_input_channels != 1 or label_manager.num_segmentation_heads != 5:
            raise ValueError(
                f"QuantENet26_5_w24 is hardcoded to in_channels=1, out_channels=5 (Dataset509_ARCADE_1x1_4c) -- "
                f"got num_input_channels={num_input_channels}, num_segmentation_heads={label_manager.num_segmentation_heads}. "
                f"Wrong dataset/plans for this trainer."
            )

        block_bits_file = os.environ.get("ENET26_5_W24_BLOCK_BITS_FILE")
        block_bits = _load_block_bits(block_bits_file)

        slope_map_file = os.environ.get("ENET26_5_W24_LEAKY_SLOPE_MAP_FILE")
        leaky_slope_map = None
        if slope_map_file:
            with open(slope_map_file) as f:
                leaky_slope_map = json.load(f)

        pretrained_checkpoint = os.environ.get("ENET26_5_W24_PRETRAINED_CHECKPOINT")
        if pretrained_checkpoint:
            return QuantENet26_5_w24.from_pretrained(
                pretrained_checkpoint, block_bits["stage_weight_bits"], block_bits["stage_act_bits"], leaky_slope_map,
            )
        return QuantENet26_5_w24(block_bits["stage_weight_bits"], block_bits["stage_act_bits"], leaky_slope_map)
