"""Trainer stub for nnUNetTrainerENet_26_9_w24_s14w12_nonneg_block's
PER-BLOCK-quantized mirror (nnunetv2.nets.QuantENet26_9_w24_s14w12_nonneg_block.
QuantENet26_9_w24_s14w12_nonneg_block) -- kept structurally parallel to
nnUNetTrainerENetQuant26_5_w24Block.py, see that file's own docstring for
the full rationale (checkpoint trainer_name resolution, real-QAT-vs-PTQ
usage, etc.), not repeated here.

ENET26_9_W24_S14W12_NONNEG_BLOCK_BLOCK_BITS_FILE (required): path to
compression/hawq/ilp_search.py's per-block output JSON ({"stage_weight_bits":
{...}, "stage_act_bits": {...}}, one entry per
QuantENet26_9_w24_s14w12_nonneg_block.BLOCK_NAMES) -- e.g.
compression/hawq/block_bits_26_9_w24_s14w12_nonneg_block_acc1x_joint.json.

ENET26_9_W24_S14W12_NONNEG_BLOCK_LEAKY_SLOPE_MAP_FILE (optional): path to
compression/post-quantization/extract_leaky_slope_map.py's output JSON
(compression/post-quantization/slope_maps/26_9_w24_s14w12_nonneg_block.json)
-- unlike QuantENet26_5_w24.py's own lossy post-hoc-averaged map, this one
is a REAL per-block scalar the FP32 checkpoint was actually trained with
(prelu_variant="nonneg_block"), so it's the safe default here, same as
S19/23_1's own recipe.

ENET26_9_W24_S14W12_NONNEG_BLOCK_PRETRAINED_CHECKPOINT (optional): path to
a real FP32 nnUNetTrainerENet_26_9_w24_s14w12_nonneg_block checkpoint to
warm-start from (conv/BN weights transferred by name+shape via
QuantENet26_9_w24_s14w12_nonneg_block.from_pretrained's own strict=False
logic) -- omit to cold-start instead.

ENET26_9_W24_S14W12_NONNEG_BLOCK_TRAINABLE_SLOPE (optional, default "1"
i.e. True): set to "0" to freeze every block's leaky-slope scalar at its
slope_map value (or QuantDecomposedLeakyAct's own default init if
unmapped) instead of letting real QAT gradients keep adapting it -- same
toggle ENET_S19_TRAINABLE_SLOPE provides for QuantENetS19Block.

ENET26_9_W24_S14W12_NONNEG_BLOCK_PRUNE_BLOCKS (optional): comma-separated
dotted block names (matching QuantENet26_9_w24_s14w12_nonneg_block.BLOCK_NAMES,
e.g. "stage3.1,regular4.1,stage3.6,stage3.3,stage3.7") to replace with
nn.Identity() via nnunetv2.nets.ENet.apply_block_pruning, applied AFTER
model construction/from_pretrained and BEFORE any state_dict load. Must be
set identically at both PTQ-calibration time
(compression/post-quantization/ptq_block.py's own --prune-blocks) and here
at inference/eval time -- the saved checkpoint's state_dict has no keys
for the pruned blocks at all, so build_network_architecture must
reconstruct the exact same pruned topology or nnUNetv2_predict's
load_state_dict fails with missing keys.

Investigation history (real HPC QAT runs on this architecture's own
acc1x_joint bit assignment got stuck at pseudo-dice 0.0 through epoch 10,
TWICE -- once with the real trained nonneg_block slope map, once with
every slope zeroed at init, both trainable_slope=True): a THIRD attempt
with no slope map at all (plain QuantReLU everywhere, no alpha) did NOT
get stuck. This points at QuantDecomposedLeakyAct's own mechanism itself
(3 chained fake-quant ops per site vs plain QuantReLU's 1, at this
architecture's own often-2-bit activation widths) rather than trainable
vs frozen vs the slope's initial value. The three env vars below let a
QAT run introduce quantization noise and slope trainability GRADUALLY
instead of both at once from epoch 0, to test that mechanism directly
without giving up the leaky activation entirely:

ENET26_9_W24_S14W12_NONNEG_BLOCK_QUANT_LEAKY_FROM_EPOCH (optional,
default 0): every QuantDecomposedLeakyAct site runs in plain FP32 (see
that class's own quant_enabled flag -- the exact algebraic identity,
alpha*x + (1-alpha)*relu(x), with NO pre_quant/act_pos/out_quant
fake-quant noise) for epochs < this value, then switches to the real
FINN-deployable quantized computation from this epoch onward. 0 (default)
means quantized from the start, i.e. no change from every other QuantENet
trainer's behavior.

ENET26_9_W24_S14W12_NONNEG_BLOCK_UNFREEZE_SLOPE_EPOCH (optional, default
0): every alpha parameter's requires_grad is False for epochs < this
value (real nn.Parameter, already in the optimizer from construction --
just excluded from receiving gradient updates until this epoch, same as
setting requires_grad=False on any parameter; no optimizer rebuild
needed) and True from this epoch onward. 0 (default) means trainable from
the start.

ENET26_9_W24_S14W12_NONNEG_BLOCK_ALPHA_LR_SCALE (optional, default 1.0):
every alpha parameter gets its own optimizer param group at
self.initial_lr * this value, weight_decay=0 (alpha is a single shared
per-block scale, not a conv/BN weight -- weight decay pulling it toward 0
has no principled justification the way it does for a conv kernel).
1.0 (default) means no change from the base trainer's single param group.

ENET26_9_W24_S14W12_NONNEG_BLOCK_ALPHA_CLAMP (optional, default unset --
no clamp): "lo,hi", e.g. "0.0,0.8" -- forward-clamps every alpha into
[lo, hi] on every forward call (see QuantDecomposedLeakyAct.set_alpha_clamp's
own docstring for the mechanism and rationale: a real nonneg_block-trained
FP32 checkpoint's own per-block slope means never exceed ~0.76 anywhere in
this network, yet alpha is currently a completely unconstrained
nn.Parameter during QAT). Applied fresh every epoch in on_epoch_start,
same idempotent-schedule pattern as the two epoch-triggered knobs above --
independent of both; can be combined with either.

ENET26_9_W24_S14W12_NONNEG_BLOCK_LEAKY_INTERNAL_BITS (optional, default
unset -- no change from current behavior): integer bit-width (e.g. "8").
When set, every QuantDecomposedLeakyAct site's own internal pre_quant/
act_pos fake-quantizers are forced to this fixed bit-width regardless of
this architecture's real, often-2-bit per-block act_bit_width -- out_quant
(the real, load-bearing boundary quantizer the next layer downstream
actually needs to match) is unchanged, always the block's real
act_bit_width. Unlike QUANT_LEAKY_FROM_EPOCH/UNFREEZE_SLOPE_EPOCH above
(per-forward-pass runtime schedules, safe to change on a resumed run),
this is a build-time architectural choice -- it changes which Brevitas
quantizer modules get constructed, not a value read fresh every forward/
epoch -- so it can only be set at model construction (here, in
build_network_architecture), not toggled mid-run via on_epoch_start the
way alpha_clamp is. Rationale: targets the exact "3 chained low-bit
fake-quantizers compound rounding error ~3x" mechanism the investigation
history above already isolated as the likely stuck-QAT cause, independent
of alpha itself. Unset (default) means every quantizer stays at
act_bit_width, i.e. no change from every other QuantENet trainer's
behavior.

ENET26_9_W24_S14W12_NONNEG_BLOCK_GUMBEL_SLOPE (optional, default "0" i.e.
off): set to "1" to enable a THIRD, orthogonal axis on top of the
investigation above -- Gumbel-Softmax categorical slope selection over the
SAME 5 fixed levels (0.0, 0.2, 0.4, 0.6, 0.8) the already-tried, already-
collapsed round-to-nearest discretization used, but chosen via a smooth,
every-level-gets-gradient-every-step relaxation instead of a hard snap
(see QuantDecomposedLeakyAct.enable_gumbel_slope's own docstring in
QuantENet.py for the full mechanism). Bare-metal test scope only: no
entropy-bonus loss term, no candidate dropout, no dedicated optimizer
param group for the logits (they land in whatever the default group is --
their name ends in ".gumbel_logits", not ".alpha", so ALPHA_LR_SCALE above
does not apply to them either).

Mutually exclusive with ALPHA_CLAMP (raises ValueError at __init__ if both
are set -- there is no continuous alpha value left to clamp once slope
selection is categorical). TRAINABLE_SLOPE and UNFREEZE_SLOPE_EPOCH both
become silent no-ops once this is enabled: TRAINABLE_SLOPE is irrelevant
(gumbel_logits is always a real trainable nn.Parameter, unconditionally --
that IS the mechanism), and UNFREEZE_SLOPE_EPOCH's own ".alpha"-suffix
filter matches nothing once every leaky site's alpha has been replaced by
gumbel_logits (no error is raised for this specific combination -- a
deliberate choice for this bare-metal pass, not an oversight).

Applied via a POST-CONSTRUCTION monkey-patch in build_network_architecture
(walks the already-built model's own .modules() and calls
enable_gumbel_slope() on every QuantDecomposedLeakyAct found), NOT
threaded through QuantENet26_9_w24_s14w12_nonneg_block's own constructor
the way TRAINABLE_SLOPE/LEAKY_INTERNAL_BITS are -- keeps this experiment's
blast radius to QuantENet.py alone, zero changes to the architecture file.
Temperature is annealed once per epoch in on_epoch_start (held at tau0=5.0
for the first 5 epochs, then exponential decay by eta=0.05/epoch toward a
tau_min=0.1 floor -- these 4 constants are hardcoded for this first pass,
not exposed as their own env vars, per the explicit bare-metal scope).
"""
import json
import math
import os

import torch
from torch import nn
from torch.optim import AdamW

from nnunetv2.nets.ENet import apply_block_pruning
from nnunetv2.nets.QuantENet import QuantDecomposedLeakyAct
from nnunetv2.nets.QuantENet26_9_w24_s14w12_nonneg_block import QuantENet26_9_w24_s14w12_nonneg_block, BLOCK_NAMES
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import nnUNetTrainerENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class _ScaledPolyLRScheduler(PolyLRScheduler):
    """Same decay curve as PolyLRScheduler, but each param_group's own
    'lr_scale' key (set at optimizer construction, default 1.0) multiplies
    the shared new_lr each step. PolyLRScheduler's OWN step() otherwise
    sets every param_group to the exact same lr value unconditionally --
    naively giving alpha's param group its own initial lr at construction
    would get silently overwritten to the main group's value on the very
    first scheduler.step() call, erasing the intended scale entirely."""
    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr * param_group.get("lr_scale", 1.0)


def _load_block_bits(path: str) -> dict:
    with open(path) as f:
        block_bits = json.load(f)
    for key in ("stage_weight_bits", "stage_act_bits"):
        missing = [b for b in BLOCK_NAMES if b not in block_bits.get(key, {})]
        if missing:
            raise ValueError(f"{path}'s {key!r} is missing entries for blocks: {missing}")
    return block_bits


# Deliberately the SAME 5 levels as the already-tried, already-collapsed
# round-to-nearest discretization experiment -- a clean, direct comparison
# of "smooth relaxation" vs. "hard snap" over the identical candidate set.
_GUMBEL_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8)
_GUMBEL_TAU0 = 5.0
_GUMBEL_WARMUP_EPOCHS = 5
_GUMBEL_ETA = 0.05
_GUMBEL_TAU_MIN = 0.1


def _gumbel_tau_for_epoch(epoch: int) -> float:
    """Held at _GUMBEL_TAU0 (near-uniform soft blend over all 5 levels) for
    the first _GUMBEL_WARMUP_EPOCHS epochs, then exponential decay toward
    _GUMBEL_TAU_MIN. Over this bare-metal test's own 3-epoch cheap-proxy
    budget, tau never leaves the warmup plateau -- this first pass is
    deliberately only testing whether the SOFT mixture phase itself avoids
    the collapse, not whether the later hardening phase also does."""
    if epoch < _GUMBEL_WARMUP_EPOCHS:
        return _GUMBEL_TAU0
    return max(_GUMBEL_TAU_MIN, _GUMBEL_TAU0 * math.exp(-_GUMBEL_ETA * (epoch - _GUMBEL_WARMUP_EPOCHS)))


class nnUNetTrainerENetQuant26_9_w24_s14w12_nonneg_blockBlock(nnUNetTrainerENet):
    def __init__(
        self, plans: dict, configuration: str, fold: int, dataset_json: dict,
        unpack_dataset: bool = True, device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        block_bits_file = os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_BLOCK_BITS_FILE")
        if not block_bits_file:
            raise ValueError(
                "ENET26_9_W24_S14W12_NONNEG_BLOCK_BLOCK_BITS_FILE must be set -- path to compression/hawq/"
                "ilp_search.py's per-block output JSON ({'stage_weight_bits': {...}, 'stage_act_bits': {...}})."
            )
        self._block_bits = _load_block_bits(block_bits_file)
        self._quant_leaky_from_epoch = int(os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_QUANT_LEAKY_FROM_EPOCH", "0"))
        self._unfreeze_slope_epoch = int(os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_UNFREEZE_SLOPE_EPOCH", "0"))
        self._alpha_lr_scale = float(os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_ALPHA_LR_SCALE", "1.0"))
        alpha_clamp_str = os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_ALPHA_CLAMP")
        self._alpha_clamp = None
        if alpha_clamp_str:
            lo_str, hi_str = alpha_clamp_str.split(",")
            self._alpha_clamp = (float(lo_str), float(hi_str))
        self._gumbel_slope = os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_GUMBEL_SLOPE", "0") != "0"
        if self._gumbel_slope and self._alpha_clamp is not None:
            raise ValueError(
                "ENET26_9_W24_S14W12_NONNEG_BLOCK_GUMBEL_SLOPE and ENET26_9_W24_S14W12_NONNEG_BLOCK_ALPHA_CLAMP "
                "are mutually exclusive -- gumbel_slope replaces alpha with a categorical pick over fixed levels, "
                "leaving no continuous value for alpha_clamp to act on."
            )

    def on_epoch_start(self):
        """Idempotent, recomputed fresh from self.current_epoch every call
        (correct across resumes too, not just a one-shot flip at some
        specific epoch boundary). See this trainer's own module docstring
        for what each of these two schedules is for."""
        for module in self.network.modules():
            if isinstance(module, QuantDecomposedLeakyAct):
                module.set_quant_enabled(self.current_epoch >= self._quant_leaky_from_epoch)
                if self._gumbel_slope:
                    module.set_gumbel_temperature(_gumbel_tau_for_epoch(self.current_epoch))
                else:
                    module.set_alpha_clamp(self._alpha_clamp)
        if self._gumbel_slope:
            act = self.network.initial.act
            if isinstance(act, QuantDecomposedLeakyAct) and act.gumbel_slope:
                entropy, argmax_level = act.gumbel_diagnostics()
                self.print_to_log_file(
                    f"[gumbel_slope] epoch {self.current_epoch}: tau={_gumbel_tau_for_epoch(self.current_epoch):.4f}, "
                    f"initial.act entropy={entropy:.4f}, argmax_level={argmax_level:.2f}"
                )
        if self._unfreeze_slope_epoch > 0:
            unfreeze = self.current_epoch >= self._unfreeze_slope_epoch
            for name, param in self.network.named_parameters():
                if name.endswith(".alpha"):
                    param.requires_grad = unfreeze
        super().on_epoch_start()

    def configure_optimizers(self):
        """Same as nnUNetTrainerENet.configure_optimizers, except alpha
        parameters get their own param group (scaled LR, zero weight
        decay) when ENET26_9_W24_S14W12_NONNEG_BLOCK_ALPHA_LR_SCALE != 1.0
        -- a no-op single-group fallback otherwise, identical to the base
        trainer's own behavior."""
        if self._alpha_lr_scale == 1.0:
            return super().configure_optimizers()
        alpha_params = [p for n, p in self.network.named_parameters() if n.endswith(".alpha")]
        other_params = [p for n, p in self.network.named_parameters() if not n.endswith(".alpha")]
        optimizer = AdamW(
            [
                {"params": other_params, "lr": self.initial_lr, "weight_decay": self.weight_decay, "lr_scale": 1.0},
                {"params": alpha_params, "lr": self.initial_lr * self._alpha_lr_scale, "weight_decay": 0.0,
                 "lr_scale": self._alpha_lr_scale},
            ],
            betas=(0.9, 0.999), eps=1e-8,
        )
        scheduler = _ScaledPolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=0.9)
        return optimizer, scheduler

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Same Brevitas parameter-scaling re-materialization quirk
        nnUNetTrainerENetQuant.py/nnUNetTrainerENetQuant26_5_w24Block.py
        already document -- only bites a resumed (--c) run."""
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
            raise ValueError("QuantENet26_9_w24_s14w12_nonneg_block is a 2D architecture. Use the nnU-Net 2d configuration.")
        label_manager = plans_manager.get_label_manager(dataset_json)
        if num_input_channels != 1 or label_manager.num_segmentation_heads != 5:
            raise ValueError(
                f"QuantENet26_9_w24_s14w12_nonneg_block is hardcoded to in_channels=1, out_channels=5 "
                f"(Dataset509_ARCADE_1x1_4c) -- got num_input_channels={num_input_channels}, "
                f"num_segmentation_heads={label_manager.num_segmentation_heads}. Wrong dataset/plans for this trainer."
            )

        block_bits_file = os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_BLOCK_BITS_FILE")
        block_bits = _load_block_bits(block_bits_file)

        slope_map_file = os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_LEAKY_SLOPE_MAP_FILE")
        leaky_slope_map = None
        if slope_map_file:
            with open(slope_map_file) as f:
                leaky_slope_map = json.load(f)

        trainable_slope = os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_TRAINABLE_SLOPE", "1") != "0"
        internal_bits_str = os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_LEAKY_INTERNAL_BITS")
        internal_bit_width = int(internal_bits_str) if internal_bits_str else None

        gumbel_slope = os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_GUMBEL_SLOPE", "0") != "0"

        pretrained_checkpoint = os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_PRETRAINED_CHECKPOINT")
        if pretrained_checkpoint:
            model = QuantENet26_9_w24_s14w12_nonneg_block.from_pretrained(
                pretrained_checkpoint, block_bits["stage_weight_bits"], block_bits["stage_act_bits"], leaky_slope_map,
                trainable_slope=trainable_slope, internal_bit_width=internal_bit_width,
            )
        else:
            model = QuantENet26_9_w24_s14w12_nonneg_block(
                block_bits["stage_weight_bits"], block_bits["stage_act_bits"], leaky_slope_map,
                trainable_slope=trainable_slope, internal_bit_width=internal_bit_width,
            )

        if gumbel_slope:
            # Post-construction monkey-patch, NOT threaded through the
            # architecture file's own constructor -- see this trainer's
            # module docstring's GUMBEL_SLOPE paragraph for why (keeps this
            # bare-metal experiment's blast radius to QuantENet.py alone).
            for module in model.modules():
                if isinstance(module, QuantDecomposedLeakyAct):
                    module.enable_gumbel_slope(levels=_GUMBEL_LEVELS, tau=_GUMBEL_TAU0)

        prune_blocks_str = os.environ.get("ENET26_9_W24_S14W12_NONNEG_BLOCK_PRUNE_BLOCKS")
        if prune_blocks_str:
            prune_names = [name.strip() for name in prune_blocks_str.split(",") if name.strip()]
            n_pruned = apply_block_pruning(model, prune_names)
            if n_pruned != len(prune_names):
                raise ValueError(
                    f"ENET26_9_W24_S14W12_NONNEG_BLOCK_PRUNE_BLOCKS={prune_blocks_str!r} named "
                    f"{len(prune_names)} block(s) but only {n_pruned} were found/pruned."
                )
        return model
