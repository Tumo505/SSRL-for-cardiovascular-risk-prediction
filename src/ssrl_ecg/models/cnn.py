from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ECGEncoder1DCNN(nn.Module):
    def __init__(self, in_ch: int = 12, width: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_ch, width, kernel_size=11, stride=2),
            ConvBlock(width, width, kernel_size=7, stride=1),
            ConvBlock(width, width * 2, kernel_size=7, stride=2),
            ConvBlock(width * 2, width * 2, kernel_size=5, stride=1),
            ConvBlock(width * 2, width * 4, kernel_size=5, stride=2),
            ConvBlock(width * 4, width * 4, kernel_size=3, stride=1),
        )
        self.out_channels = width * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class SSLReconstructionModel(nn.Module):
    def __init__(self, in_ch: int = 12, width: int = 64):
        super().__init__()
        self.encoder = ECGEncoder1DCNN(in_ch=in_ch, width=width)
        c = self.encoder.out_channels
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(c, c // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(c // 2, c // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(c // 4, c // 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c // 8, in_ch, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        # Keep exact input length after transpose-conv.
        if x_hat.shape[-1] > x.shape[-1]:
            x_hat = x_hat[..., : x.shape[-1]]
        elif x_hat.shape[-1] < x.shape[-1]:
            pad = x.shape[-1] - x_hat.shape[-1]
            x_hat = torch.nn.functional.pad(x_hat, (0, pad))
        return x_hat


class ECGClassifier(nn.Module):
    def __init__(self, encoder: ECGEncoder1DCNN, n_classes: int = 5):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.encoder.out_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        pooled = self.pool(z).squeeze(-1)
        return self.head(pooled)
