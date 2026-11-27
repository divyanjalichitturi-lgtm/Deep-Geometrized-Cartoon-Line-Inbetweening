"""Compact two-branch geometry-aware encoder/decoder for line inbetweening."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels, max(4, out_channels // 8), 1), nn.SiLU(), nn.Conv2d(max(4, out_channels // 8), out_channels, 1), nn.Sigmoid())

    def forward(self, x):
        y = self.body(x)
        return F.silu((y * self.attention(y)) + self.skip(x))


class FusionGate(nn.Module):
    """Learn per-location how much geometry should influence raster features."""
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())
        self.project = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, raster, geometry):
        gate = self.gate(torch.cat([raster, geometry], dim=1))
        return self.project(torch.cat([raster, geometry * gate], dim=1))


class GeometryAwareDualBranchNet(nn.Module):
    """Inputs: line A, vertices A, graph A, line B, vertices B, graph B, t.

    Outputs a grayscale line-intensity map where 1 denotes a dark line. The
    raster and ML240 geometry streams are encoded separately, then fused at
    three scales so topology contributes without replacing image evidence.
    """
    input_channels = 7
    output_channels = 1
    architecture = "Geometry-Aware Dual-Branch Residual U-Net"

    def __init__(self, base_channels: int = 24):
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.raster_stem = ResidualBlock(3, widths[0])
        self.geometry_stem = ResidualBlock(4, widths[0])
        self.raster_down = nn.ModuleList([ResidualBlock(widths[i], widths[i + 1]) for i in range(3)])
        self.geometry_down = nn.ModuleList([ResidualBlock(widths[i], widths[i + 1]) for i in range(3)])
        self.fuse = nn.ModuleList([FusionGate(w) for w in widths])
        self.pool = nn.AvgPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[i + 1], widths[i], 2, 2) for i in (2, 1, 0)])
        self.decode = nn.ModuleList([ResidualBlock(widths[i] * 2, widths[i]) for i in (2, 1, 0)])
        self.head = nn.Sequential(nn.Conv2d(widths[0], widths[0] // 2, 3, padding=1), nn.SiLU(inplace=True), nn.Conv2d(widths[0] // 2, 1, 1))

    def forward(self, x):
        raster = torch.cat([x[:, 0:1], x[:, 3:4], x[:, 6:7]], 1)
        geometry = torch.cat([x[:, 1:3], x[:, 4:6]], 1)
        r, g = self.raster_stem(raster), self.geometry_stem(geometry)
        fused = [self.fuse[0](r, g)]
        for i in range(3):
            r = self.raster_down[i](self.pool(r))
            g = self.geometry_down[i](self.pool(g))
            fused.append(self.fuse[i + 1](r, g))
        y = fused[-1]
        for j, i in enumerate((2, 1, 0)):
            y = self.up[j](y)
            skip = fused[i]
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = self.decode[j](torch.cat([y, skip], 1))
        return torch.sigmoid(self.head(y))


# Kept as an import alias for existing project scripts.
GeometryAwareUNet = GeometryAwareDualBranchNet


def create_model(device=None):
    model = GeometryAwareDualBranchNet()
    return model.to(device) if device is not None else model

