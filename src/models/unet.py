import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, inChannels: int, outChannels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UnetZ500(nn.Module):
    """
    UNet that is shape-safe for your grid (H=12, W=64) by using ONLY 2 pooling steps.

    Input:  (B, 1, H, W)
    Output: (B, 1, H, W)

    If useResidual=True, model predicts the future field as:
        out = x + f(x)
    """

    def __init__(
        self,
        inChannels: int = 1,
        outChannels: int = 1,
        baseChannels: int = 32,
        useResidual: bool = True,
    ):
        super().__init__()
        self.useResidual = useResidual

        # Encoder (2 levels)
        self.enc1 = ConvBlock(inChannels, baseChannels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(baseChannels, baseChannels * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(baseChannels * 2, baseChannels * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(
            baseChannels * 4, baseChannels * 2, kernel_size=2, stride=2
        )
        self.dec2 = ConvBlock(baseChannels * 4, baseChannels * 2)

        self.up1 = nn.ConvTranspose2d(
            baseChannels * 2, baseChannels, kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock(baseChannels * 2, baseChannels)

        self.outConv = nn.Conv2d(baseChannels, outChannels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xIn = x

        # Down
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Up
        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        out = self.outConv(d1)

        if self.useResidual:
            out = out + xIn

        return out