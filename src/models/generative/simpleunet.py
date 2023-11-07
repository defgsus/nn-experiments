from typing import Tuple, Optional

import torch
import torch.nn as nn

from src.models.encoder.number_embedding import SinusoidalNumberEmbedding


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t):
        # 第一次卷积
        h = self.bnorm1(self.relu(self.conv1(x)))
        # 时间嵌入
        time_emb = self.relu(self.time_mlp(t))
        # 扩展到最后2个维度
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # 添加时间通道
        h = h + time_emb
        # 第二次卷积
        h = self.bnorm2(self.relu(self.conv2(h)))
        # 上采样或者下采样
        return self.transform(h)


class SimpleUnet(nn.Module):
    """
    Unet架构的一个简化版本
    """
    def __init__(
            self,
            image_channels: int = 3,
            down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024),
            up_channels: Tuple[int, ...] = (1024, 512, 256, 128, 64),
            time_emb_dim: int = 32,
            code_dim: Optional[int] = None,
    ):
        super().__init__()

        big_dim = time_emb_dim
        if code_dim is not None:
            big_dim += code_dim
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalNumberEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        if code_dim is not None:
            self.code_mlp = nn.Sequential(
                nn.Linear(code_dim, code_dim),
                nn.ReLU()
            )

        # 初始预估
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # 下采样
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], big_dim)
            for i in range(len(down_channels)-1)
        ])
        # 上采样
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], big_dim, up=True)
            for i in range(len(up_channels)-1)
        ])

        self.output = nn.Conv2d(up_channels[-1], image_channels, 1)

    def forward(self, x, timestep, code: Optional[torch.Tensor] = None):
        # 时间嵌入
        t = self.time_mlp(timestep)

        if code is not None:
            if not hasattr(self, "code_mlp"):
                raise ValueError(f"code specified in forward but no code_dim in constructor")
            t = torch.concat([t, self.code_mlp(code)], dim=-1)

        # 初始卷积
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # 添加残差结构作为额外的通道
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)
