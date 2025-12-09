"""
Lightweight CBAM and ASFF implementations for experiments.

CBAM: Convolutional Block Attention Module (channel + spatial attention)
ASFF: Adaptively Spatial Feature Fusion (simplified variant)

These are written to be easy to integrate in a runtime-patch: import the modules
and use their forward() on appropriate feature maps. This file doesn't modify
Ultralytics internals directly — use helper code to insert these modules where
needed in the model's forward pass.
"""
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

    def forward(self, x):
        # x: (B,C,H,W)
        b, c, _, _ = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        maxp = F.adaptive_max_pool2d(x, 1).view(b, c)
        out = self.mlp(avg) + self.mlp(maxp)
        out = torch.sigmoid(out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # x: (B,C,H,W)
        avg = torch.mean(x, dim=1, keepdim=True)
        maxp, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, maxp], dim=1)
        out = torch.sigmoid(self.conv(cat))
        return x * out


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class ASFF(nn.Module):
    """A simplified ASFF: fuse 3 feature maps into a target resolution.

    Inputs are a list of 3 tensors [x_small, x_mid, x_large] where their spatial
    sizes differ by powers of 2. The module resizes them to the middle size and
    computes a learnable weighted sum (softmax over learned weights), then a conv.
    """

    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()
        assert len(in_channels) == 3, 'Expect three feature maps'
        self.in_convs = nn.ModuleList([nn.Conv2d(ic, out_channels, 1) for ic in in_channels])
        # weights for three feature maps
        self.weight = nn.Parameter(torch.zeros(3))
        self.fuse_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, feats: List[torch.Tensor]):
        # feats: list of 3 tensors with shapes (B,C_i,H_i,W_i)
        # we'll target the spatial size of the middle map (feats[1])
        target = feats[1]
        _, _, Ht, Wt = target.shape
        outs = []
        for i, f in enumerate(feats):
            x = self.in_convs[i](f)
            if x.shape[2:] != (Ht, Wt):
                x = F.interpolate(x, size=(Ht, Wt), mode='bilinear', align_corners=False)
            outs.append(x)
        w = torch.softmax(self.weight, dim=0)
        fused = w[0] * outs[0] + w[1] * outs[1] + w[2] * outs[2]
        out = self.fuse_conv(fused)
        return out


if __name__ == '__main__':
    # quick sanity check
    x1 = torch.randn(1, 128, 80, 80)
    x2 = torch.randn(1, 256, 40, 40)
    x3 = torch.randn(1, 512, 20, 20)
    cb = CBAM(256)
    y = cb(x2)
    asff = ASFF([128, 256, 512], out_channels=256)
    z = asff([x1, x2, x3])
    print('CBAM out', y.shape, 'ASFF out', z.shape)