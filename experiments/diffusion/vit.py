from typing import Optional

import torch
import torch.nn as nn
import torchvision.models

from src.models.util import ResidualAdd


class ViT(nn.Module):
    def __init__(
            self,
            image_size: int,
            image_channels: int,
            patch_size: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            output_channels: Optional[int] = None,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
    ):
        super().__init__()
        if output_channels is None:
            output_channels = image_channels

        model = torchvision.models.VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.patch_size = model.patch_size
        self.proj_in = nn.Conv2d(image_channels, hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.transformer = model.encoder

        self.proj_out = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=self.patch_size, stride=self.patch_size),
            # refine the patches
            nn.Sequential(
                *[
                    ResidualAdd(nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                        nn.GELU(),
                    ))
                    for _ in range(patch_size // 2)
                ]
            ),
            nn.Conv2d(hidden_dim, output_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        y = self.proj_in(x)
        shape = y.shape
        y = y.flatten(-2).permute(0, 2, 1)

        # add (fake) expected class token
        y = torch.cat([y, y[:, :1, :]], dim=1)

        y = self.transformer(y)
        y = y[:, :-1, :].permute(0, 2, 1)
        y = self.proj_out(y.view(shape))
        return y
