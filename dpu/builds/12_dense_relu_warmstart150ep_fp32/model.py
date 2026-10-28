"""S12 dense ReLU warm-start (150ep) float ENet for the Vitis AI flow.

Uses the repo's own plain-PyTorch ENet (vendored as ENet.py, no Brevitas) with
the exact trainer settings from nas/slurm/stage_12_dense_relu_warmstart150ep.job.
The Vitis AI container is Python 3.7, where typing.Literal does not exist, so it
is aliased from typing_extensions before ENet.py is imported.
"""
import glob
import os
import sys
import typing

import numpy as np
import torch
from PIL import Image

import typing_extensions

if not hasattr(typing, "Literal"):
    typing.Literal = typing_extensions.Literal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ENet import ENet  # noqa: E402

CHECKPOINT = os.environ.get("CHECKPOINT_PATH", "/workspace/inputs/checkpoint.pth")
CALIB_IMAGES = "/workspace/inputs/calib"
LABELS = "/workspace/inputs/labels"


def input_shape():
    return tuple(int(v) for v in os.environ["INPUT_SHAPE"].split(","))


# Assumed supported by the board's DPU (3.0 release notes), but vai_q_pytorch 3.0 crashes
# quantizing nndct_argmax_dim ("'tuple' object has no attribute 'values'" in do_scan), and the
# Inspector rejects it for the generic DPUCZDX8G_ISA1_B4096 target. Quantize without it.
WITH_ARGMAX = False



class ArgmaxHead(torch.nn.Module):
    """Class map [N,1,H,W] instead of logits [N,C,H,W]."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.argmax(self.net(x), dim=1, keepdim=True)


def build_model() -> torch.nn.Module:
    net = ENet(
        in_channels=1,
        out_channels=5,
        channels=(4, 16, 32, 16, 4),
        bottlenecks_per_stage=(4, 8, 8, 2, 1),
        decoder_type="upsample_conv",
        use_dilated=True,
        use_asymmetric=False,
        use_strided=True,
        use_dsc=False,
        context_pattern="dense_dilation",
        use_prelu=False,
    )
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    net.load_state_dict(ckpt["network_weights"], strict=True)
    net.eval()
    return ArgmaxHead(net).eval() if WITH_ARGMAX else net


def _zscore(path):
    # nnU-Net ZScoreNormalization over the whole image, as in
    # deployment/img_intf/utils/preprocess.py (minus its FINN-specific uint8 step:
    # the DPU applies its own input fixpoint scale).
    img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    img = (img - img.mean()) / max(float(img.std()), 1e-8)
    return torch.from_numpy(img)[None, None]


def _images(part):
    """Disjoint, deterministic split of the available images: calib / eval / rest."""
    files = sorted(glob.glob(os.path.join(CALIB_IMAGES, "*_0000.png")))
    return [f for i, f in enumerate(files) if (i % 3 == 0) == (part == "calib")]


def calibration_batches(limit: int = 100):
    for f in _images("calib")[:limit]:
        yield _zscore(f)


def eval_batches(limit: int = 200):
    """Yield (input, label[H,W] int64) from images disjoint from the calibration set."""
    for f in _images("eval")[:limit]:
        case = os.path.basename(f)[:-len("_0000.png")]
        lab = np.asarray(Image.open(os.path.join(LABELS, case + ".png")))
        yield _zscore(f), torch.from_numpy(lab.astype(np.int64))

