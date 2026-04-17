from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class FusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class RGBDUNet(nn.Module):
    """Dual-branch U-Net for RGB-only and RGB-D experiments."""

    def __init__(
        self,
        *,
        num_classes: int,
        base_channels: int = 32,
        use_depth: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_depth = use_depth
        widths = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

        self.rgb_stem = ConvBlock(3, widths[0], dropout=dropout)
        self.rgb_down1 = EncoderBlock(widths[0], widths[1], dropout=dropout)
        self.rgb_down2 = EncoderBlock(widths[1], widths[2], dropout=dropout)
        self.rgb_down3 = EncoderBlock(widths[2], widths[3], dropout=dropout)
        self.rgb_down4 = EncoderBlock(widths[3], widths[4], dropout=dropout)

        if self.use_depth:
            self.depth_stem = ConvBlock(1, widths[0], dropout=dropout)
            self.depth_down1 = EncoderBlock(widths[0], widths[1], dropout=dropout)
            self.depth_down2 = EncoderBlock(widths[1], widths[2], dropout=dropout)
            self.depth_down3 = EncoderBlock(widths[2], widths[3], dropout=dropout)
            self.depth_down4 = EncoderBlock(widths[3], widths[4], dropout=dropout)
        else:
            self.depth_stem = None
            self.depth_down1 = None
            self.depth_down2 = None
            self.depth_down3 = None
            self.depth_down4 = None

        factor = 2 if self.use_depth else 1
        self.fuse1 = FusionBlock(widths[0] * factor, widths[0])
        self.fuse2 = FusionBlock(widths[1] * factor, widths[1])
        self.fuse3 = FusionBlock(widths[2] * factor, widths[2])
        self.fuse4 = FusionBlock(widths[3] * factor, widths[3])
        self.fuse_bottleneck = FusionBlock(widths[4] * factor, widths[4])

        self.up4 = DecoderBlock(widths[4], widths[3], widths[3], dropout=dropout)
        self.up3 = DecoderBlock(widths[3], widths[2], widths[2], dropout=dropout)
        self.up2 = DecoderBlock(widths[2], widths[1], widths[1], dropout=dropout)
        self.up1 = DecoderBlock(widths[1], widths[0], widths[0], dropout=dropout)
        self.head = nn.Conv2d(widths[0], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _fuse(self, rgb_feature: torch.Tensor, depth_feature: torch.Tensor | None, fusion_block: FusionBlock) -> torch.Tensor:
        if self.use_depth:
            if depth_feature is None:
                raise ValueError("Depth input is required when use_depth=True.")
            merged = torch.cat([rgb_feature, depth_feature], dim=1)
        else:
            merged = rgb_feature
        return fusion_block(merged)

    def forward(self, image: torch.Tensor, depth: torch.Tensor | None = None) -> torch.Tensor:
        rgb1 = self.rgb_stem(image)
        rgb2 = self.rgb_down1(rgb1)
        rgb3 = self.rgb_down2(rgb2)
        rgb4 = self.rgb_down3(rgb3)
        rgb5 = self.rgb_down4(rgb4)

        if self.use_depth:
            if depth is None:
                raise ValueError("Depth tensor must be passed to RGBDUNet when use_depth=True.")
            depth1 = self.depth_stem(depth)
            depth2 = self.depth_down1(depth1)
            depth3 = self.depth_down2(depth2)
            depth4 = self.depth_down3(depth3)
            depth5 = self.depth_down4(depth4)
        else:
            depth1 = depth2 = depth3 = depth4 = depth5 = None

        skip1 = self._fuse(rgb1, depth1, self.fuse1)
        skip2 = self._fuse(rgb2, depth2, self.fuse2)
        skip3 = self._fuse(rgb3, depth3, self.fuse3)
        skip4 = self._fuse(rgb4, depth4, self.fuse4)
        bottleneck = self._fuse(rgb5, depth5, self.fuse_bottleneck)

        x = self.up4(bottleneck, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        return self.head(x)
