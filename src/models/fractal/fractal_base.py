from typing import Union, Iterable, Optional, Literal, List

import torch
import torch.nn as nn


class FractalBaseLayer(nn.Module):

    def __init__(
            self,
            num_channels: int,
            axis: int,
            scale: Union[None, float, Iterable[float], torch.Tensor] = None,
            offset: Union[None, Iterable[float], torch.Tensor] = None,
            mixer: Union[None, torch.Tensor, List[List[float]]] = None,
            learn_mixer: bool = False,
            learn_scale: bool = False,
            learn_offset: bool = False,
    ):
        """
        A base nn layer for calculating fractals.

        Think of the input as coordinates with `channels` dimensions
        and output some transformed version of it.

        The module accepts any shape as long as the `axis` dimension has the size `channels`

        :param num_channels: int, number of channels
        :param axis: int, axis of the channels
        :param learn_param: do train the parameter
        :param mixer: matrix of shape (len(param), len(param)) to transform the final values
        """
        super().__init__()
        self.num_channels = num_channels
        self.axis = axis

        self.mixer = None
        if mixer is None and learn_mixer:
            mixer = torch.diag(torch.Tensor([1] * self.num_channels))

        if mixer is not None:
            if isinstance(mixer, torch.Tensor):
                mixer = torch.Tensor(mixer)
            if mixer.shape != torch.Size((self.num_channels, self.num_channels)):
                raise ValueError(f"Expected `mixer` to have shape ({self.num_channels}, {self.num_channels}), got {mixer.shape}")
            self.mixer = nn.Parameter(mixer, requires_grad=learn_mixer)

        self.scale = None
        if scale is None and learn_scale:
            scale = 1.
        if scale is not None:
            if isinstance(scale, (int, float)):
                scale = torch.Tensor([scale] * self.num_channels)
            elif not isinstance(scale, torch.Tensor):
                scale = torch.Tensor(scale)
            if scale.shape != torch.Size((self.num_channels, )):
                raise ValueError(f"Expected `scale` to have shape ({self.num_channels}, ), got {scale.shape}")
            self.scale = nn.Parameter(scale, requires_grad=learn_scale)

        self.offset = None
        if offset is None and learn_offset:
            offset = [0.] * self.num_channels
        if offset is not None:
            if not isinstance(offset, torch.Tensor):
                offset = torch.Tensor(offset)
            if offset.shape != torch.Size((self.num_channels, )):
                raise ValueError(f"Expected `offset` to have shape ({self.num_channels}, ), got {offset.shape}")
            self.offset = nn.Parameter(offset, requires_grad=learn_offset)

    def fractal(self, coords: torch.Tensor, axis: int) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ndim = x.ndim
        axis = self.axis
        if axis < 0:
            axis = ndim + self.axis

        scale = self.scale
        offset = self.offset
        slices = [None] * (ndim - axis - 1)
        if scale is not None:
            scale = scale[..., *slices]
        if offset is not None:
            offset = offset[..., *slices]

        if scale is not None or offset is not None:
            slices = [slice(None, None)] * self.axis
            x = x.clone()

        if scale is not None:
            x[*slices] *= scale
        if offset is not None:
            x[*slices] = x[*slices] + offset

        output = self.fractal(x, axis)

        if self.mixer is not None:
            perm_dims = list(range(output.ndim))
            perm_dims.append(perm_dims.pop(axis))
            output = output.permute(*perm_dims) @ self.mixer
            perm_dims = list(range(output.ndim))
            perm_dims.insert(axis, perm_dims.pop(-1))
            output = output.permute(*perm_dims)

        return output
