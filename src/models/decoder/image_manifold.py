from collections import OrderedDict
from typing import Optional, Tuple, Union, Iterable

import torch
import torch.nn as nn

from src.algo.space2d import Space2d


class ResidualLinearBlock(nn.Module):
    def __init__(
            self,
            num_hidden: int,
            num_layers: int,
            batch_norm: bool = True,
            concat: bool = False,
    ):
        super().__init__()
        self.do_concat = concat
        self.layers = nn.Sequential(OrderedDict([
            *(
                (("norm", nn.BatchNorm1d(num_hidden)), ) if batch_norm else tuple()
            ),
            *(
                (f"layer_{i + 1}", nn.Sequential(OrderedDict([
                    ("linear", nn.Linear(num_hidden, num_hidden)),
                    ("act", nn.GELU()),
                ])))
                for i in range(num_layers)
            )
        ]))

    def forward(self, x):
        if self.do_concat:
            return torch.concat([x, self.layers(x)], dim=-1)
        else:
            return x + self.layers(x)

    def extra_repr(self):
        return "concat=True" if self.do_concat else ""


class ImageManifoldDecoder(nn.Module):

    def __init__(
            self,
            num_input_channels: int,
            num_output_channels: int = 3,
            num_hidden: int = 256,
            num_blocks: int = 2,
            num_layers_per_block: int = 2,
            concat_residual: Union[bool, Iterable[bool]] = False,
            pos_embedding_freqs: Iterable[float] = (7, 17),
            batch_norm: bool = True,
            default_shape: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.default_shape = default_shape
        self.num_output_channels = num_output_channels
        self.pos_embedding_freqs = tuple(pos_embedding_freqs)
        self.pos_embedding_size = (len(self.pos_embedding_freqs) + 1) * 2
        if isinstance(concat_residual, bool):
            self.concat_residual = (concat_residual, ) * num_blocks
        else:
            self.concat_residual = tuple(concat_residual)
            if len(concat_residual) != num_blocks:
                raise ValueError(f"len(concat_residual) must be {num_blocks}, got {len(self.concat_residual)}")

        hidden_sizes = [num_hidden]
        hs = num_hidden
        for i, concat in enumerate(self.concat_residual):
            if concat:
                hs *= 2
            hidden_sizes.append(hs)

        self.pos_to_color = nn.Sequential(OrderedDict([
            ("linear_in", nn.Linear(num_input_channels + self.pos_embedding_size, num_hidden)),
            ("act_in", nn.GELU()),
            ("resblocks", nn.Sequential(OrderedDict([
                (
                    f"resblock_{i+1}",
                    ResidualLinearBlock(
                        num_hidden=hs,
                        num_layers=num_layers_per_block,
                        batch_norm=batch_norm,
                        concat=concat,
                    )
                )
                for i, (concat, hs) in enumerate(zip(self.concat_residual, hidden_sizes))
            ]))),
            ("linear_out", nn.Linear(hidden_sizes[-1], num_output_channels)),
            ("act_out", nn.Sigmoid()),
        ]))
        self._cur_space = None
        self._cur_space_shape = None

    def extra_repr(self):
        args = [
            f"pos_embedding_freqs={self.pos_embedding_freqs}",
            f"concat_residual={self.concat_residual}",
        ]
        if self.default_shape is not None:
            args.append(f"default_shape={self.default_shape}")
        return ", ".join(args)

    def forward(self, x: torch.Tensor, shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if x.ndim not in (1, 2):
            raise ValueError(f"Expecting ndim 1 or 2, got {x.shape}")

        if x.ndim == 2:
            return torch.concat([
                self.forward(x_i, shape).unsqueeze(0)
                for x_i in x
            ])

        space, shape = self.get_pos_embedding(shape)

        codes = torch.concat([
            x.unsqueeze(0).expand(space.shape[0], x.shape[-1]),
            space
        ], dim=1)
        return self.pos_to_color(codes).permute(1, 0).view(self.num_output_channels, *shape)

    def get_pos_embedding(self, shape: Optional[Tuple[int, int]] = None):
        if shape is None:
            shape = self.default_shape
        if shape is None:
            raise ValueError("Must either define `default_shape` or `shape`")

        if shape != self._cur_space_shape:
            space = Space2d(shape=(2, *shape)).space().to(self.pos_to_color[0].weight)
            space = space.permute(1, 2, 0).view(-1, 2)
            space = torch.concat([
                space,
                *(
                    (space * freq).sin()
                    for freq in self.pos_embedding_freqs
                )
            ], 1)
            self._cur_space = space.to(self.pos_to_color[0].weight)
            self._cur_space_shape = shape

        return self._cur_space, shape

    def weight_images(self, **kwargs):
        images = []
        for i, p in enumerate(self.parameters()):
            if p.ndim == 2 and any(s > 1 for s in p.shape):
                images.append(p)

        return images


class ImageManifoldEncoder(nn.Module):
    """NOT REALLY TESTED YET"""
    def __init__(
            self,
            num_output_channels: int,
            num_input_channels: int = 3,
            num_hidden: int = 256,
            num_blocks: int = 2,
            num_layers_per_block: int = 2,
    ):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.color_to_pos = nn.Sequential(OrderedDict([
            ("linear_in", nn.Linear(num_input_channels + 2, num_hidden)),
            ("act_in", nn.GELU()),
            ("resblocks", nn.Sequential(OrderedDict([
                (f"resblock_{i+1}", ResidualLinearBlock(num_hidden=num_hidden, num_layers=num_layers_per_block))
                for i in range(num_blocks)
            ]))),
            ("linear_out", nn.Linear(num_hidden, num_output_channels)),
        ]))
        self._cur_space = None
        self._cur_space_shape = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (3, 4):
            raise ValueError(f"Expecting ndim 3 or 4, got {x.shape}")

        if x.ndim == 4:
            return torch.concat([
                self.forward(x_i).unsqueeze(0)
                for x_i in x
            ])

        space = self.get_space(x.shape[-2:])
        codes = torch.concat([x, space], dim=0).flatten(1).permute(1, 0)
        return self.color_to_pos(codes).permute(1, 0).mean(-1)

    def get_space(self, shape: Tuple[int, int] = None):
        if shape != self._cur_space_shape:
            space = Space2d(shape=(2, *shape)).space().to(self.color_to_pos[0].weight)
            self._cur_space = space
            self._cur_space_shape = shape

        return self._cur_space

    def weight_images(self, **kwargs):
        images = []
        for i, p in enumerate(self.parameters()):
            if p.ndim == 2:
                images.append(p)

        return images
