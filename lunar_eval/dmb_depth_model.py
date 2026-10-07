"""DMBNet-style dual-branch encoder with MSA and boundary-refined depth head.

Adapted from the DMBNet recipe (frozen VFM + trainable encoder, multiscale
adapter, boundary auxiliary branch) to monocular dense depth regression on
lunar NAC orthophotos. The frozen DINOv3 branch consumes PRECOMPUTED
features (see precompute_dinov3_features.py); only the conv stem, MSA,
decoder, and heads are trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# DINOv3-S: 4 stages x 384 channels at H/16 (512 -> 32x32 per stage).
DINO_CHANNELS = [384, 384, 384, 384]
DINO_SIZE = 32  # spatial size of precomputed maps at 512px input


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__()
        p = k // 2
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CBAM(nn.Module):
    """Lightweight channel+spatial attention (per DMBNet's MSA)."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hid = max(channels // reduction, 8)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid, channels, 1), nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid(),
        )

    def forward(self, x):
        x = x * self.channel(x)
        mx = x.amax(dim=1, keepdim=True)
        avg = x.mean(dim=1, keepdim=True)
        return x * self.spatial(torch.cat([mx, avg], dim=1))


class MSA(nn.Module):
    """Multiscale Adapter: parallel 3/5/7/9 convs + CBAM, with residual."""

    def __init__(self, channels):
        super().__init__()
        self.reduce = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.branches = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(channels, channels, k, padding=k // 2, bias=False),
                           nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
             for k in (3, 5, 7, 9)])
        self.merge = nn.Conv2d(channels * 4, channels, 1, bias=False)
        self.attn = CBAM(channels)

    def forward(self, frozen_feat, train_feat):
        f = self.reduce(torch.cat([frozen_feat, train_feat], dim=1))
        fused = torch.cat([b(f) for b in self.branches], dim=1)
        out = self.attn(self.merge(fused) + f)
        return out


class TrainableStem(nn.Module):
    """Lightweight 4-stage conv encoder on the raw orthophoto (1 channel)."""

    def __init__(self, widths=(32, 64, 128, 256)):
        super().__init__()
        c = list(widths)
        self.stage1 = nn.Sequential(ConvBlock(1, c[0]), ConvBlock(c[0], c[0]))
        self.stage2 = nn.Sequential(ConvBlock(c[0], c[1], s=2), ConvBlock(c[1], c[1]))
        self.stage3 = nn.Sequential(ConvBlock(c[1], c[2], s=2), ConvBlock(c[2], c[2]))
        self.stage4 = nn.Sequential(ConvBlock(c[2], c[3], s=2), ConvBlock(c[3], c[3]))

    def forward(self, x):
        s1 = self.stage1(x)                       # H
        s2 = self.stage2(s1)                      # H/2
        s3 = self.stage3(s2)                      # H/4
        s4 = self.stage4(s3)                      # H/8
        return [s1, s2, s3, s4]


class DecoderBlock(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2, bias=False)
        self.fuse = nn.Sequential(
            ConvBlock(cout + skip, cout), ConvBlock(cout, cout))

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class BAM(nn.Module):
    """Boundary Auxiliary Module: boundary logits gate depth features."""

    def __init__(self, channels):
        super().__init__()
        self.c1b = nn.Conv2d(channels, channels, 1, bias=False)
        self.c1m = nn.Conv2d(channels, channels, 1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, mask_feat, boundary_feat):
        gated = torch.sigmoid(self.c1b(boundary_feat)) * self.c1m(mask_feat)
        return self.refine(gated + self.c1m(mask_feat))


class DMBDepth(nn.Module):
    def __init__(self, dino_channels=DINO_CHANNELS, dino_size=DINO_SIZE,
                 widths=(32, 64, 128, 256)):
        super().__init__()
        self.dino_size = dino_size
        w = list(widths)
        self.stem = TrainableStem(w)
        # Project frozen DINO features and trainable features to a common width
        # at each stage, then MSA-fuse. DINO maps are at dino_size (H/16); the
        # trainable stages are pooled/interpolated to match before fusion, and
        # the fused representation lives at the coarsest-common resolution.
        self.frozen_proj = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(dc, w[3], 1, bias=False), nn.BatchNorm2d(w[3]),
                           nn.ReLU(inplace=True)) for dc in dino_channels])
        self.train_proj = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(w[i], w[3], 1, bias=False), nn.BatchNorm2d(w[3]),
                           nn.ReLU(inplace=True)) for i in range(4)])
        self.msa = nn.ModuleList([MSA(w[3]) for _ in range(4)])

        # Progressive decoder: fused features at H/16 are upsampled and fused
        # with the trainable stem features (projected to decoder widths).
        # Progressive decoder widths at H/8, H/4, H/2, H. Skip connections
        # come from the trainable stem stages, projected to the decoder width.
        dc = [128, 64, 32, 32]
        self.skip_proj = nn.ModuleList([
            nn.Conv2d(w[3], dc[0], 1, bias=False),  # stages[3] (H/8)  -> 128
            nn.Conv2d(w[2], dc[1], 1, bias=False),  # stages[2] (H/4)  -> 64
            nn.Conv2d(w[1], dc[2], 1, bias=False),  # stages[1] (H/2)  -> 32
            nn.Conv2d(w[0], dc[3], 1, bias=False),  # stages[0] (H)    -> 32
        ])
        self.dec3 = DecoderBlock(w[3], dc[0], dc[0])   # H/16 -> H/8 (+s3)
        self.dec2 = DecoderBlock(dc[0], dc[1], dc[1])  # H/8  -> H/4 (+s2)
        self.dec1 = DecoderBlock(dc[1], dc[2], dc[2])  # H/4  -> H/2  (+s1)
        self.dec0 = DecoderBlock(dc[2], dc[3], dc[3])  # H/2  -> H    (+s0)

        feat = dc[3]
        self.mask_head = nn.Sequential(
            ConvBlock(feat, feat), nn.Conv2d(feat, feat, 3, padding=1), nn.ReLU(inplace=True))
        self.boundary_head = nn.Conv2d(feat, 1, 1)
        self.bam = BAM(feat)
        self.depth_head = nn.Conv2d(feat, 1, 1)

    def forward(self, image, dino_feats):
        """
        image: (B, 1, H, W) orthophoto tile, values ~[0, 255].
        dino_feats: list of 4 (B, 384, dino_size, dino_size) frozen features.
        Returns (depth, boundary_logits) at (B, 1, H, W).
        """
        stages = self.stem(image)  # [s1..s4] at H, H/2, H/4, H/8
        # Coarse fused representation at dino_size (H/16), 4 stages aggregated.
        fused = 0
        for i in range(4):
            f = self.frozen_proj[i](dino_feats[i])           # (B,w3,S,S)
            t = self.train_proj[i](stages[i])
            t = F.adaptive_avg_pool2d(t, self.dino_size)      # (B,w3,S,S)
            fused = fused + self.msa[i](f, t)
        fused = fused / 4.0

        d3 = self.dec3(fused, self.skip_proj[0](stages[3]))
        d2 = self.dec2(d3, self.skip_proj[1](stages[2]))
        d1 = self.dec1(d2, self.skip_proj[2](stages[1]))
        d0 = self.dec0(d1, self.skip_proj[3](stages[0]))
        # dec0 upsamples 2x -> H; ensure exact size
        if d0.shape[-2:] != image.shape[-2:]:
            d0 = F.interpolate(d0, size=image.shape[-2:], mode="bilinear",
                               align_corners=False)

        mask_feat = self.mask_head(d0)
        boundary_logits = self.boundary_head(mask_feat)
        refined = self.bam(mask_feat, mask_feat)
        depth = self.depth_head(refined)
        return depth, boundary_logits


def boundary_target(dtm, mask, top_frac=0.15):
    """Boundary label: pixels in the top `top_frac` of valid DTM gradient."""
    B = dtm.shape[0]
    targets = torch.zeros_like(dtm)
    for i in range(B):
        m = mask[i, 0]
        if m.sum() < 100:
            continue
        d = dtm[i, 0]
        gx = torch.zeros_like(d)
        gy = torch.zeros_like(d)
        gx[:, 1:-1] = (d[:, 2:] - d[:, :-2]) / 2
        gy[1:-1, :] = (d[2:, :] - d[:-2, :]) / 2
        grad = torch.sqrt(gx * gx + gy * gy)
        grad = torch.where(m, grad, torch.zeros_like(grad))
        k = max(int(top_frac * m.sum().item()), 1)
        thresh = torch.topk(grad.flatten(), k).values[-1]
        targets[i, 0] = ((grad >= thresh) & m).float()
    return targets


def gradient_loss(pred, target, mask):
    """L1 on spatial gradients, encouraging sharp rims/ridges."""
    loss = 0.0
    for dim in (-1, -2):
        dp = torch.diff(pred, dim=dim)
        dt = torch.diff(target, dim=dim)
        m = (mask & torch.roll(mask, shifts=-1, dims=dim))
        m = m.narrow(dim, 0, dp.shape[dim])
        if m.sum() > 0:
            loss = loss + (torch.abs(dp - dt) * m).sum() / m.sum()
    return loss / 2.0
