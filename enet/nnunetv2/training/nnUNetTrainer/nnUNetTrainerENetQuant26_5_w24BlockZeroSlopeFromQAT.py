"""Variant of nnUNetTrainerENetQuant26_5_w24Block for one specific
experiment in the stuck-training investigation (see
nnUNetTrainerENetQuant26_9_w24_s14w12_nonneg_blockBlock.py's own module
docstring for the fuller history): does warm-starting the leaky-slope
mechanism from an ALREADY quantization-adapted checkpoint (real QAT-trained,
not just FP32) avoid the collapse a fresh FP32->QAT+leaky transition seems
to trigger?

ENET26_5_W24_ZEROSLOPE_QAT_CHECKPOINT (required): path to a real, already
QAT-TRAINED (not FP32) nnUNetTrainerENetQuant26_5_w24Block checkpoint --
e.g. nnUNetTrainerENetQuant26_5_w24Block_26_5_w24_qat_block_bits's own
checkpoint_final.pth (dice=0.6806, trained with block_bits_26_5_w24.json,
FROZEN slope from a "standard"-prelu lossy-averaged map -- see
QuantENet26_5_w24.py's own module docstring). Loaded the same way
QuantENet26_5_w24.from_pretrained does (name+shape match, strict=False),
EXCEPT every key containing ".alpha" or ".one_minus_alpha" is explicitly
EXCLUDED from the transfer first. This matters: that source checkpoint's
own alpha/one_minus_alpha are already real, meaningful (if lossy) FROZEN
buffers from its own training -- a naive from_pretrained-style transfer
would silently pull those old frozen values into this run's own alpha
parameter (same key name, same shape, since state_dict() doesn't
distinguish nn.Parameter from a registered buffer), silently overriding
the fresh zero-init this experiment actually wants. Every OTHER key
(conv/BN weights, already adapted to real quantization noise over that
checkpoint's own full QAT run) transfers normally.

ENET26_5_W24_ZEROSLOPE_LEAKY_SLOPE_MAP_FILE (required): path to a slope
map with the same 23 block-name keys as compression/post-quantization/
slope_maps/26_5_w24.json (the lossy per-channel-PReLU-averaged map) --
e.g. compression/post-quantization/slope_maps/26_5_w24_zeroed.json (every
value replaced with 0.0). trainable_slope=True always in this trainer
(hardcoded, not an env var -- zero-init only makes sense paired with
letting it move).

ENET26_5_W24_ZEROSLOPE_BLOCK_BITS_FILE (required): path to compression/
hawq/ilp_search.py's per-block output JSON, same schema as
nnUNetTrainerENetQuant26_5_w24Block's own ENET26_5_W24_BLOCK_BITS_FILE --
e.g. compression/hawq/block_bits_26_5_w24.json, the same bit assignment
the source QAT checkpoint was itself trained with (this experiment's
whole point is isolating the leaky-slope mechanism's own effect, not also
changing the bit assignment at the same time -- pass a different file
deliberately if a future run wants to combine both changes).
"""
import json
import os

import torch
from torch import nn

from nnunetv2.nets.QuantENet26_5_w24 import QuantENet26_5_w24, BLOCK_NAMES
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import nnUNetTrainerENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

_EXCLUDE_SUBSTRINGS = (".alpha", ".one_minus_alpha")


def _load_block_bits(path: str) -> dict:
    with open(path) as f:
        block_bits = json.load(f)
    for key in ("stage_weight_bits", "stage_act_bits"):
        missing = [b for b in BLOCK_NAMES if b not in block_bits.get(key, {})]
        if missing:
            raise ValueError(f"{path}'s {key!r} is missing entries for blocks: {missing}")
    return block_bits


class nnUNetTrainerENetQuant26_5_w24BlockZeroSlopeFromQAT(nnUNetTrainerENet):
    def __init__(
        self, plans: dict, configuration: str, fold: int, dataset_json: dict,
        unpack_dataset: bool = True, device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        qat_checkpoint = os.environ.get("ENET26_5_W24_ZEROSLOPE_QAT_CHECKPOINT")
        if not qat_checkpoint:
            raise ValueError(
                "ENET26_5_W24_ZEROSLOPE_QAT_CHECKPOINT must be set -- path to an already QAT-trained "
                "nnUNetTrainerENetQuant26_5_w24Block checkpoint (NOT an FP32 one) to warm-start from, with "
                "alpha/one_minus_alpha explicitly excluded from the transfer."
            )
        block_bits_file = os.environ.get("ENET26_5_W24_ZEROSLOPE_BLOCK_BITS_FILE")
        if not block_bits_file:
            raise ValueError(
                "ENET26_5_W24_ZEROSLOPE_BLOCK_BITS_FILE must be set -- path to compression/hawq/"
                "ilp_search.py's per-block output JSON ({'stage_weight_bits': {...}, 'stage_act_bits': {...}})."
            )
        self._block_bits = _load_block_bits(block_bits_file)

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Same Brevitas parameter-scaling re-materialization quirk every
        other QuantENet trainer already documents -- only bites a resumed
        (--c) run."""
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

        block_bits_file = os.environ.get("ENET26_5_W24_ZEROSLOPE_BLOCK_BITS_FILE")
        block_bits = _load_block_bits(block_bits_file)

        slope_map_file = os.environ.get("ENET26_5_W24_ZEROSLOPE_LEAKY_SLOPE_MAP_FILE")
        if not slope_map_file:
            raise ValueError(
                "ENET26_5_W24_ZEROSLOPE_LEAKY_SLOPE_MAP_FILE must be set -- path to a slope map JSON with "
                "QuantENet26_5_w24's own 23 encoder/context block-name keys."
            )
        with open(slope_map_file) as f:
            leaky_slope_map = json.load(f)

        qat_checkpoint_path = os.environ.get("ENET26_5_W24_ZEROSLOPE_QAT_CHECKPOINT")
        model = QuantENet26_5_w24(
            block_bits["stage_weight_bits"], block_bits["stage_act_bits"], leaky_slope_map,
        )

        checkpoint = torch.load(qat_checkpoint_path, map_location="cpu", weights_only=False)
        source_state_dict = checkpoint["network_weights"]
        model_state_dict = model.state_dict()
        transferable = {
            key: value for key, value in source_state_dict.items()
            if key in model_state_dict and model_state_dict[key].shape == value.shape
            and not any(s in key for s in _EXCLUDE_SUBSTRINGS)
        }
        n_excluded_present = sum(
            1 for key in source_state_dict
            if any(s in key for s in _EXCLUDE_SUBSTRINGS) and key in model_state_dict
        )
        missing, unexpected = model.load_state_dict(transferable, strict=False)
        assert not unexpected, f"unexpected keys after strict=False load (should be impossible): {unexpected}"
        print(
            f"QuantENet26_5_w24 warm-started from QAT checkpoint {qat_checkpoint_path} "
            f"(alpha/one_minus_alpha EXCLUDED from transfer): transferred {len(transferable)}/"
            f"{len(model_state_dict)} model keys, {n_excluded_present} alpha/one_minus_alpha keys "
            f"deliberately skipped (left at slope_map's own zero-init instead), "
            f"{len(missing) - n_excluded_present} other keys left uninitialized (expected Brevitas-only "
            f"quantizer params)."
        )
        return model
