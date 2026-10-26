"""QAT-ready S12 ENet: functional ops (cat / add / interpolate) become nn.Modules.

QatProcessor.trainable_model requires every quantized op to be an instance of a
pytorch_nndct functional module, and rejects several ops sharing one parent module
name. The vendored ENet.py is left untouched (the PTQ flow uses it as-is); the
forward methods of the affected blocks are swapped here at runtime. The weights and
the maths are identical, so the FP32 checkpoint loads unchanged.
"""
import torch
import torch.nn.functional as F
from pytorch_nndct.nn import DeQuantStub, QuantStub
from pytorch_nndct.nn.modules import functional as nf

from model import build_model


def _initial_forward(self, x):
    # ReLU per branch (relu(cat) == cat(relu)) so each one fuses into its conv on the DPU.
    c = self.act_conv(self.bn_conv(self.conv(x)))
    p = self.act_pool(self.bn_pool(self.pool_proj(self.pool(x))))
    return self.q_cat([c, p], dim=1)


def _split_initial_bn(block):
    """BN over cat([conv, pool]) -> a conv+BN pair per branch (QAT cannot quantize a standalone BN).

    The pooled branch gets an identity 1x1 conv so its BN can be folded and trained; the
    function is unchanged because BN is per-channel.
    """
    n_conv = block.conv.out_channels
    n_pool = block.conv.in_channels
    bn = block.bn

    def part(sl):
        new = torch.nn.BatchNorm2d(sl.stop - sl.start, eps=bn.eps, momentum=bn.momentum)
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(new, name).data.copy_(getattr(bn, name).data[sl])
        return new

    block.bn_conv = part(slice(0, n_conv))
    block.bn_pool = part(slice(n_conv, n_conv + n_pool))
    block.pool_proj = torch.nn.Conv2d(n_pool, n_pool, 1, bias=False)
    block.pool_proj.weight.data.copy_(torch.eye(n_pool).view(n_pool, n_pool, 1, 1))
    block.act_conv = torch.nn.ReLU()
    block.act_pool = torch.nn.ReLU()
    del block.bn
    del block.act


def _residual_forward(self, x):
    out = self.reduce(x)
    out = self.conv_bn_act(out)
    out = self.dropout(self.expand(out))
    return self.out_act(self.q_add(x, out))


def _down_forward(self, x):
    input_size = x.size()
    main = self.pool(x)
    pad_channels = self.out_channels - self.in_channels
    if pad_channels > 0:
        padding = torch.zeros(main.shape[0], pad_channels, main.shape[2], main.shape[3],
                              dtype=main.dtype, device=main.device)
        main = self.q_cat([main, padding], dim=1)
    elif pad_channels < 0:
        main = main[:, : self.out_channels]
    out = self.reduce(x)
    out = self.conv(out)
    out = self.dropout(self.expand(out))
    return self.out_act(self.q_add(main, out)), None, input_size


def _up_forward(self, x, output_size, indices=None):
    assert indices is None, "only the upsample_conv decoder is supported"
    main = self.main_proj(x)
    if self.learned_skip_upsample:
        main = self.main_upsample(main)
    elif self.skip_interp_mode == "nearest":
        main = self.q_interp(main, size=output_size[2:], mode="nearest")
    else:
        main = self.q_interp(main, size=output_size[2:], mode="bilinear", align_corners=False)
    if self.skip_resize_conv is not None:
        main = self.skip_resize_conv(main)
    out = self.reduce(x)
    out = self.up(out)
    out = self.dropout(self.expand(out))
    return self.out_act(self.q_add(main, out))


def _enet_forward(self, x):
    x = self.quant_stub(x)
    x = self.initial(x)
    x, _, size1 = self.down1(x)
    x = self.regular1(x)
    x, _, size2 = self.down2(x)
    x = self.stage2(x)
    x = self.proj2_to_3(x)
    x = self.stage3(x)
    x = self.up4(x, size2, None)
    x = self.regular4(x)
    x = self.up5(x, size1, None)
    x = self.regular5(x)
    x = self.final(x)
    return self.dequant_stub(x)


_PATCHES = {
    "ENet": (_enet_forward, {"quant_stub": QuantStub, "dequant_stub": DeQuantStub}),
    "InitialBlock": (_initial_forward, {"q_cat": nf.Cat}),
    "RegularBottleneck": (_residual_forward, {"q_add": nf.Add}),
    "RegularNoProjectionBottleneck": (_residual_forward, {"q_add": nf.Add}),
    "DownsamplingBottleneck": (_down_forward, {"q_cat": nf.Cat, "q_add": nf.Add}),
    "UpsamplingBottleneck": (_up_forward, {"q_add": nf.Add, "q_interp": nf.Interpolate}),
}


def make_qat_ready(net):
    for module in list(net.modules()):
        name = type(module).__name__
        if name not in _PATCHES:
            continue
        forward, extras = _PATCHES[name]
        for attr, cls in extras.items():
            setattr(module, attr, cls())
        if name == "InitialBlock":
            _split_initial_bn(module)
        if name == "DownsamplingBottleneck":
            module.in_channels = module.reduce[0].in_channels if not isinstance(module.reduce[0], torch.nn.MaxPool2d) \
                else module.reduce[1].in_channels
        # same class name keeps the traced node names identical to the PTQ graph
        module.__class__ = type(name, (type(module),), {"forward": forward})
    return net


def build_qat_model():
    return make_qat_ready(build_model())
