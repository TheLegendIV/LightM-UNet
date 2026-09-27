"""BSCNet, cherry-picked from https://github.com/njupt-quanzhou/BSCNet
(mmseg/models/backbones/bscnet.py -- a BiSeNet/STDC-style dual-branch
network with boundary-aware feature fusion, config at configs/bscnet/
bscnet.py). The upstream repo is a full mmsegmentation fork; the backbone
file only uses mmcv for a handful of thin wrappers (BatchNorm2d selection,
kaiming/constant init, the @BACKBONES registry decorator, pretrained-
checkpoint loading) that are stripped here rather than pulling in the full
mmcv/mmseg dependency stack. Computational graph is otherwise verbatim.

The upstream repo wires the backbone's last output into a separate mmseg
`FCNHead` (config: in_channels=128, channels=64, num_convs=1, kernel_size=3,
dropout_ratio=0) via mmseg's EncoderDecoder, plus 3 auxiliary deep-
supervision heads (boundary + two intermediate features) used only during
their own training recipe. `BSCNet` below reimplements just that main
FCNHead shape directly (no mmseg dependency) and drops the auxiliary heads
entirely -- consistent with how the other cherry-picked baselines in this
repo (TVResNet101UNet, PUNet, UNeXtS) are wrapped: main segmentation output
only, no deep supervision.

Also, like UNeXt-S's `encoder1`, the upstream `conv1` hardcodes 3 input
channels regardless of any constructor argument; here it actually takes
`in_channels` so this works for ARCADE's single-channel grayscale input.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d  # upstream defaults to nn.SyncBatchNorm for multi-GPU; single-GPU here
bn_mom = 0.1


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, groups=planes, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, groups=planes, stride=1, padding=1, bias=False)
        self.bn3 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn4 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        out = self.bn4(self.conv4(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class newelppm(nn.Module):
    def __init__(self, inplanes, outplanes):
        super().__init__()
        self.dim = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                                  BatchNorm2d(outplanes, momentum=bn_mom), nn.ReLU(inplace=True))
        self.pool8 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2))
        self.pool4 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4))
        self.pool2 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8))
        self.pool1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

        self.process2 = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, kernel_size=3, groups=outplanes, padding=1, bias=False),
            BatchNorm2d(outplanes, momentum=bn_mom), nn.ReLU(inplace=True))
        self.process4 = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, kernel_size=3, groups=outplanes, padding=1, bias=False),
            BatchNorm2d(outplanes, momentum=bn_mom), nn.ReLU(inplace=True))
        self.process8 = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, kernel_size=3, groups=outplanes, padding=1, bias=False),
            BatchNorm2d(outplanes, momentum=bn_mom),
            nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(outplanes, momentum=bn_mom), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.dim(x)
        pool_list = [self.pool2(x), self.pool4(x), self.pool8(x)]

        x_1 = F.interpolate(self.pool1(x), size=(pool_list[0].shape[-2], pool_list[0].shape[-1]), mode="bilinear")
        x_2 = F.interpolate(self.process2(x_1 + pool_list[0]), size=(pool_list[1].shape[-2], pool_list[1].shape[-1]),
                             mode="bilinear")
        x_4 = F.interpolate(self.process4(x_2 + pool_list[1]), size=(pool_list[2].shape[-2], pool_list[2].shape[-1]),
                             mode="bilinear")
        x_8 = F.interpolate(self.process8(x_4 + pool_list[2]), size=(x.shape[-2], x.shape[-1]), mode="bilinear")

        return x_8 + x


class SeBFM2(nn.Module):
    """Semantic bilateral feature-map fusion between a low-res and a high-res branch."""

    def __init__(self, l_inplanes=64, h_inplanes=128, l_outplanes=64, h_outplanes=128):
        super().__init__()
        self.l2h = nn.Sequential(nn.Conv2d(l_inplanes, h_outplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                  BatchNorm2d(h_outplanes, momentum=bn_mom), nn.ReLU(inplace=True))
        self.h_seg = nn.Sequential(
            nn.Conv2d(h_outplanes, h_outplanes, kernel_size=3, groups=h_outplanes, padding=1, bias=False),
            BatchNorm2d(h_outplanes, momentum=bn_mom),
            nn.Conv2d(h_outplanes, h_outplanes, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(h_outplanes, momentum=bn_mom), nn.ReLU(inplace=True))

        self.h2l = nn.Sequential(nn.Conv2d(h_inplanes, l_outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                                  BatchNorm2d(l_outplanes, momentum=bn_mom), nn.ReLU(inplace=True))
        self.l_seg = nn.Sequential(
            nn.Conv2d(l_outplanes, l_outplanes, kernel_size=3, groups=l_outplanes, padding=1, bias=False),
            BatchNorm2d(l_outplanes, momentum=bn_mom),
            nn.Conv2d(l_outplanes, l_outplanes, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(l_outplanes, momentum=bn_mom), nn.ReLU(inplace=True))

    def forward(self, x):
        # x[0] = low-resolution branch, x[1] = high-resolution branch
        batch_l, channel_l, height_l, width_l = x[0].size()
        batch_h, channel_h, height_h, width_h = x[1].size()

        x_h2l = F.interpolate(self.h2l(x[1]), size=(height_l, width_l), mode="bilinear")
        l_out = self.l_seg(x_h2l + x[0])

        x_l2h = F.interpolate(self.l2h(x[0]), size=(height_h, width_h), mode="bilinear")
        h_out = self.h_seg(x_l2h + x[1])

        return [l_out, h_out]


class BAFM_CFF(nn.Module):
    """Boundary-aware fusion: channel-attention fusion of deep + shallow features."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                    BatchNorm2d(128, momentum=bn_mom), nn.ReLU(inplace=True))
        self.dw3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, groups=128, padding=1, bias=False),
                                  BatchNorm2d(128, momentum=bn_mom),
                                  nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  BatchNorm2d(128, momentum=bn_mom), nn.ReLU(inplace=True))

    def forward(self, x):
        # x[0] = deep semantic feature, x[1] = shallow spatial feature
        out = self.conv1(torch.cat((x[0], x[1]), dim=1))
        gate = torch.sigmoid(F.adaptive_avg_pool2d(out, (1, 1)))
        return out * gate


class BAFM_BRB(nn.Module):
    """Boundary-guided directional running-average refinement between boundary pixels.

    Verbatim port of the upstream per-pixel loop (only runs at 1/64 of input
    resolution -- 1/8 from the backbone's own downsampling, another 1/8
    applied here -- so the Python-level loop is over an 8x8 grid for a
    512x512 input, not the full resolution)."""

    def forward(self, x_f):
        x_f0 = F.interpolate(x_f[0], scale_factor=1 / 8)
        x_f1 = F.interpolate(x_f[1], scale_factor=1 / 8)

        xh_lr, xh_rl = x_f0.clone(), x_f0.clone()
        xv_tb, xv_bt = x_f0.clone(), x_f0.clone()
        N, _, h, w = x_f1.shape

        for i in range(N):
            for row in range(h):
                index = (x_f1[i, 0, row, :] == 0).nonzero()
                if index.numel() == 0:
                    continue
                for j in range(len(index) - 1):
                    start, end = index[j], index[j + 1]
                    for temp in range(start + 1, end):
                        xh_lr[i, 0, row, temp] = x_f0[i, 0, row, start:temp + 1].mean()
                        xh_rl[i, 0, row, temp] = x_f0[i, 0, row, temp:end + 1].mean()

        for i in range(N):
            for col in range(w):
                index = (x_f1[i, 0, :, col] == 0).nonzero()
                if index.numel() == 0:
                    continue
                for j in range(len(index) - 1):
                    start, end = index[j], index[j + 1]
                    for temp in range(start + 1, end):
                        xv_tb[i, 0, temp, col] = x_f0[i, 0, start:temp + 1, col].mean()
                        xv_bt[i, 0, temp, col] = x_f0[i, 0, temp:end + 1, col].mean()

        return F.interpolate(xh_lr + xh_rl + xv_tb + xv_bt, scale_factor=8)


class BSCNetBackbone(nn.Module):
    def __init__(self, in_channels=3, block=Bottleneck, layers=(2, 2, 2, 2), planes=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom), nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(block, planes, planes, layers[0], stride=2)
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)

        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)
        self.layer5 = self._make_layer(block, planes * 8, planes * 16, 1, stride=2)

        self.layer3_ = self._make_layer(block, planes * 2, planes * 2, 2)
        self.layer4_ = self._make_layer(block, planes * 2, planes * 2, 2)
        self.layer5_ = self._make_layer(block, planes * 2, planes * 4, 1)

        self.elppm = newelppm(planes * 16, planes * 4)

        self.SeBFM1 = SeBFM2(l_inplanes=128, h_inplanes=64, l_outplanes=128, h_outplanes=64)
        self.SeBFM2 = SeBFM2(l_inplanes=256, h_inplanes=64, l_outplanes=256, h_outplanes=64)

        self.boundary_head = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=0)

        self.bafm_cff = BAFM_CFF()
        self.bafm_brb = BAFM_BRB()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        if stride == 2:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, groups=inplanes, bias=False, padding=1),
                BatchNorm2d(inplanes, momentum=bn_mom),
                nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
                BatchNorm2d(planes, momentum=bn_mom))
        elif inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
                                        BatchNorm2d(planes, momentum=bn_mom))
        else:
            downsample = None

        layers = [block(inplanes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(block(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        height_output = x.shape[-2] // 8
        width_output = x.shape[-1] // 8

        x = self.conv1(x)
        x = self.layer1(x)
        x_s = self.layer2(x)

        xl = self.layer3(x_s)
        xh = self.layer3_(x_s)
        xl, xh = self.SeBFM1([xl, xh])

        xl = self.layer4(xl)
        xh = self.layer4_(xh)
        xl, xh = self.SeBFM2([xl, xh])

        xl = self.layer5(xl)
        xl = F.interpolate(self.elppm(xl), size=[height_output, width_output], mode="bilinear")
        xh = self.layer5_(xh)

        x_backbone = xl + xh

        x_bound = F.interpolate(self.boundary_head(x_backbone), size=[height_output, width_output], mode="bilinear")
        x_bound_ch1 = torch.sigmoid(x_bound[:, 1:2, :, :])
        with torch.no_grad():
            x_bound_mask = (x_bound_ch1 > 0.1).float()

        x_f = self.bafm_cff([x_backbone, x_s])
        x_out = self.bafm_brb([x_f, x_bound_mask])

        return x_out + x_backbone  # 128 channels, at 1/8 input resolution


class BSCNet(nn.Module):
    """Backbone + the upstream config's main FCNHead shape (in=128, mid=64),
    no auxiliary/deep-supervision heads -- see module docstring."""

    def __init__(self, in_channels=3, num_classes=19, layers=(2, 2, 2, 2), planes=32, head_planes=64):
        super().__init__()
        self.backbone = BSCNetBackbone(in_channels=in_channels, layers=layers, planes=planes)
        self.head = nn.Sequential(
            nn.Conv2d(planes * 4, head_planes, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(head_planes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(head_planes, num_classes, kernel_size=1))

    def forward(self, x):
        input_size = x.shape[-2:]
        feat = self.backbone(x)
        logits = self.head(feat)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
