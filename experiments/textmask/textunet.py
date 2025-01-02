import math
import unittest
from functools import partial
from typing import Union, Callable, Type, Dict, List, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.util import activation_to_module, normalization_to_module
from src.models.attention import Attention1d
from src.models.cnn import CheapConv1d, CheapConvTranspose1d
from src.util.params import param_make_list
from src.util.module import dump_module_stacktrace
from src.util.embedding import create_diagonal_matrix


class TextUnetLayer(nn.Module):
    def __init__(
            self,
            num_channels_in: int,
            num_channels_out: int,
            kernel_size: int,
            padding: Union[int, str] = 0,
            stride: int = 1,
            dilation: int = 1,
            pool: Union[None, Tuple[int, int]] = None,  # (kernel-size, stride)
            norm: Union[None, str, Type[nn.Module]] = None,
            activation: Union[None, str, Callable] = None,
            cheap: bool = False,
            transpose: bool = False,
    ):
        super().__init__()
        self._transpose = transpose

        if isinstance(padding, str):
            if padding == "same":
                padding = (kernel_size - 1) // 2 * dilation
            else:
                raise ValueError(f"Unsupported padding '{padding}'")

        self.norm = normalization_to_module(norm, channels=num_channels_in)

        if transpose:
            conv_class = CheapConvTranspose1d if cheap else nn.ConvTranspose1d
        else:
            conv_class = CheapConv1d if cheap else nn.Conv1d

        self.conv = conv_class(
            num_channels_in,
            num_channels_out,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
        )

        self.pooling = None
        if pool is not None:
            if transpose:
                self.pooling = nn.MaxUnpool1d(
                    kernel_size=pool[0],
                    stride=pool[1],
                )
            else:
                self.pooling = nn.MaxPool1d(
                    kernel_size=pool[0],
                    stride=pool[1],
                    return_indices=True,
                    ceil_mode=True,
                )

        self.act = activation_to_module(activation)

    def forward(
            self,
            x: torch.Tensor,
            pool_indices: Optional[torch.Tensor] = None,
            output_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if self.norm is not None:
            x = self.norm(x)

        if not self._transpose:  # forward path

            x = self.conv(x)

            if self.act is not None:
                x = self.act(x)

            indices = None
            if self.pooling is not None:
                x, indices = self.pooling(x)

            return x, indices

        else:  # transposed path

            if self.pooling is not None:
                assert pool_indices is not None
                x = self.pooling(x, pool_indices)

            x = self.conv(x)

            if output_size is not None and x.shape[-1] != output_size:
                x = F.pad(x, (0, output_size - x.shape[-1]))

            if self.act is not None:
                x = self.act(x)

            return x


class TextUnet(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            num_channels: Union[int, Iterable[int]],
            kernel_size: Union[int, Iterable[int]] = 3,
            padding: Union[int, str, Iterable[Union[int, str]]] = 0,   # can be "same"
            stride: Union[int, Iterable[int]] = 1,
            dilation: Union[int, Iterable[int]] = 1,
            pool: Union[None, Tuple[int, int], Iterable[Union[None, Tuple[int, int]]]] = None,  # (kernel-size, stride)
            norm: Union[None, str, Type[nn.Module]] = None,
            activation: Union[None, str, Callable] = None,
            cheap: Union[bool, Iterable[bool]] = False,
            diagonal_embedding: bool = True,
            symmetric_embedding: bool = True,
    ):
        super().__init__()

        num_channels_ = param_make_list(num_channels, num_layers, "num_channels")
        kernel_size_ = param_make_list(kernel_size, num_layers, "kernel_size")
        padding_ = param_make_list(padding, num_layers, "padding")
        stride_ = param_make_list(stride, num_layers, "stride")
        dilation_ = param_make_list(dilation, num_layers, "dilation")
        pool_ = param_make_list(pool, num_layers, "pool", arg_is_tuple=True)
        cheap_ = param_make_list(cheap, num_layers, "cheap")

        self.embedding = nn.Embedding(vocab_size, num_channels_[0])
        if diagonal_embedding:
            with torch.no_grad():
                self.embedding.weight[:] = create_diagonal_matrix(self.embedding.weight.shape)

        self.encoder = nn.ModuleDict()
        self.decoder = nn.ModuleDict()

        for i in range(num_layers):
            is_last_layer = i == num_layers - 1
            self.encoder.add_module(
                f"layer_{i + 1}",
                TextUnetLayer(
                    num_channels_in=num_channels_[i],
                    num_channels_out=num_channels_[min(i + 1, num_layers - 1)],
                    kernel_size=kernel_size_[i],
                    padding=padding_[i],
                    stride=stride_[i],
                    dilation=dilation_[i],
                    pool=None if pool_[i] is "None" else pool_[i],
                    norm=None if is_last_layer else norm,
                    activation=activation,
                    cheap=cheap_[i],
                )
            )
            self.decoder.add_module(
                f"layer_{i + 1}",
                TextUnetLayer(
                    num_channels_in=num_channels_[-max(1, i)],
                    num_channels_out=num_channels_[-(i + 1)],
                    kernel_size=kernel_size_[-(i + 1)],
                    padding=padding_[-(i + 1)],
                    stride=stride_[-(i + 1)],
                    dilation=dilation_[-(i + 1)],
                    pool=None if pool_[-(i + 1)] is "None" else pool_[-(i + 1)],
                    norm=None if is_last_layer else norm,
                    activation=activation,
                    cheap=cheap_[i],
                    transpose=True,
                )
            )

        self.head = nn.Linear(num_channels_[-1], vocab_size, bias=False)
        if symmetric_embedding:
            self.head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)  # B,L,C
        state = x.permute(0, 2, 1)  # B,C,L

        state_history = [state]
        pool_history = [None]
        for layer in self.encoder.values():
            state, indices = layer(state)
            state_history.append(state)
            pool_history.append(indices)

        for i, layer in enumerate(self.decoder.values()):
            if i > 0:
                state = state + state_history[-(i + 1)]

            state = layer(
                state,
                pool_history[-(i + 1)],
                state_history[-(i + 2)].shape[-1],
            )

        state = state.permute(0, 2, 1)  # B,L,C

        logits = self.head(state)  # B,L,V

        return logits


class TestTextUnet(unittest.TestCase):

    @torch.no_grad()
    def test_unet(self):
        from src.util import iter_parameter_permutations

        for params in iter_parameter_permutations({
            "num_channels": [32, [32, 48, 64]],
            "kernel_size": [1, 2, 3],
            "stride": [1, 2, [3, 2, 1]],
            "padding": [0, 1, [2, 3, 4], "same"],
            "pool": [None, (3, 2), [None, (2, 1), None]],
            "shape": [
                (3, 300),
            ],
            "cheap": [False, True],
        }):
            msg = ", ".join(f"{key}={repr(value)}" for key, value in params.items())
            shape = params["shape"]

            model = TextUnet(
                vocab_size=256,
                num_layers=3,
                num_channels=params["num_channels"],
                kernel_size=params["kernel_size"],
                stride=params["stride"],
                padding=params["padding"],
                pool=params["pool"],
                cheap=params["cheap"],
            ).eval()

            input = torch.randint(0, 255, shape)

            try:
                print()
                dump_module_stacktrace(model, input)
            except:
                print(model)
                print(msg)
                raise

            output = model(input).argmax(-1)

            self.assertEqual(
                input.shape,
                output.shape,
                msg,
            )
