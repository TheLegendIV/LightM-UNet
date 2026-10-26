"""Plain-PyTorch model for the Vitis AI quantizer/inspector (no Brevitas).

Do not import LayerQuantEnetFINN here: its quantizers and FINN-specific
substitutions are not meant for the Vitis AI flow. Re-create the same topology
with ordinary nn.Conv2d / BatchNorm / ReLU / pooling / upsampling modules and
transfer the trained weights explicitly. Keep the layer order DPU-friendly
(CONV -> BN -> ReLU).
"""
import os

import torch


def input_shape():
    return tuple(int(v) for v in os.environ["INPUT_SHAPE"].split(","))


def build_model() -> torch.nn.Module:
    """Return the float model in eval mode with the trained weights loaded."""
    # TODO:
    #   1. build the plain-PyTorch network
    #   2. load /workspace/inputs/checkpoint.pth and map parameter names
    #   3. compare its outputs against the original implementation on real
    #      images before trusting any quantization result
    raise NotImplementedError


def calibration_batches(limit: int = 200):
    """Yield float input tensors shaped like input_shape().

    Must use the same preprocessing as deployment (see
    deployment/img_intf/utils/preprocess.py).
    """
    # TODO: read /workspace/inputs/calib/*
    raise NotImplementedError
