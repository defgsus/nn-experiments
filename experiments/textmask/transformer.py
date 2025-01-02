import math
import unittest
from functools import partial
from typing import Union, Callable, Type, Dict, List, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.util import activation_to_module, normalization_to_module
from src.models.attention import Attention1d
from src.models.cnn import CheapConv1d
from src.util.params import param_make_tuple
from src.util.embedding import create_diagonal_matrix


class ConvTextLayer(nn.Module):
    def __init__(
            self,
            num_channels_in: int,
            num_channels_out: int,
            kernel_size: int = 7,
            dilation: int = 1,
            activation: Union[None, str, Callable] = "gelu",
            cheap: bool = False,
    ):
        super().__init__()

        padding = (kernel_size - 1) // 2 * dilation
        self.conv = (CheapConv1d if cheap else nn.Conv1d)(
            num_channels_in,
            num_channels_out,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.act = activation_to_module(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)

        if self.act is not None:
            y = self.act(y)

        if x.shape == y.shape:
            y = y + x

        return y


class ConvTextBlock(nn.Module):
    def __init__(
            self,
            num_layers: int,
            num_channels: int,
            kernel_size: Union[int, Iterable[int]] = 7,
            dilation: Union[int, Iterable[int]] = 1,
            norm: Union[None, str, Type[nn.Module]] = None,
            activation: Union[None, str, Callable] = "gelu",
            residual: bool = True,
            cheap: Union[bool, Iterable[bool]] = False,
            attention: Optional[str] = None,  # "QKin", "QK", "QV", "KV", "QKV"
            attention_activation: Union[None, str, Callable] = "elu+1",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.residual = residual
        self.attention_type = attention

        self.layers = nn.ModuleList()
        for i, kernel_size_, dilation_, cheap_ in zip(
                range(num_layers),
                param_make_tuple(kernel_size, num_layers, "kernel_size"),
                param_make_tuple(dilation, num_layers, "dilation"),
                param_make_tuple(cheap, num_layers, "cheap"),
        ):
            is_last = i == num_layers - 1
            is_attention = is_last and attention and attention != "QKin"

            self.layers.add_module(
                f"layer_{i + 1}",
                ConvTextLayer(
                    num_channels_in=num_channels,
                    num_channels_out=num_channels * (len(attention) if is_attention else 1),
                    kernel_size=kernel_size_,
                    dilation=dilation_,
                    cheap=cheap_,
                    activation=None if attention and is_last else activation,
                )
            )

        self.attention = None
        if attention:
            self.attention = Attention1d(
                activation=attention_activation,
            )

        self.act = None
        if attention:
            self.act = activation_to_module(activation)

        self.norm = normalization_to_module(norm, channels=num_channels)

    def extra_repr(self) -> str:
        text = f"residual={self.residual}"
        if self.attention_type:
            text = (
                f"{text}, attention='{self.attention_type}'"
            )
        return text

    def forward(self, x: torch.Tensor, query: Optional[torch.Tensor] = None, key: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_x = x

        #if self.norm is not None:
        #    x = self.norm(x)

        prev_x = x
        for layer in self.layers:
            x = layer(x)
            #x, prev_x = layer(x), x

        if self.attention is not None:
            if self.attention_type == "QKin":
                assert query is not None and key is not None
                x = self.attention(query, key, x.permute(0, 2, 1))

            else:
                # channel-wise attention (put channels into last dim)
                source = x.permute(0, 2, 1)
                x = prev_x.permute(0, 2, 1)

                if self.attention_type in ("QK", "KQ"):
                    q, k = torch.split(source, source.shape[-1] // 2, dim=-1)
                    x = self.attention(q, k, x)
                elif self.attention_type in ("QV", "VQ"):
                    q, v = torch.split(source, source.shape[-1] // 2, dim=-1)
                    x = self.attention(q, x, v)
                elif self.attention_type in ("KV", "VK"):
                    k, v = torch.split(source, source.shape[-1] // 2, dim=-1)
                    x = self.attention(x, k, v)
                elif self.attention_type in ("QKV", "QVK", "KQV", "KVQ", "VQK", "VKQ"):
                    q, k, v = torch.split(source, source.shape[-1] // 3, dim=-1)
                    x = self.attention(q, k, v)
                else:
                    raise AssertionError(f"Invalid `attention` '{self.attention_type}'")

            x = x.permute(0, 2, 1)

        if self.act is not None:
            x = self.act(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.residual:
            x = x + original_x

        return x


class ConvTextTransformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            encoder_blocks: Iterable[ConvTextBlock],
            decoder_blocks: Optional[Iterable[ConvTextBlock]] = None,
            diagonal_embedding: bool = True,
            symmetric_embedding: bool = True,
    ):
        super().__init__()

        encoder_blocks_list = nn.ModuleDict()
        first_block = None
        for i, block in enumerate(encoder_blocks):
            encoder_blocks_list.add_module(f"block_{i + 1}", block)
            if first_block is None:
                first_block = block

        num_channels = first_block.num_channels

        self.embedding = nn.Embedding(vocab_size, num_channels)
        if diagonal_embedding:
            with torch.no_grad():
                self.embedding.weight[:] = create_diagonal_matrix(self.embedding.weight.shape)

        self.encoder_blocks = encoder_blocks_list

        self.decoder_blocks = None
        if decoder_blocks is not None:
            self.qk = CheapConv1d(num_channels, num_channels * 2, kernel_size=3, padding=1)

            self.decoder_blocks = nn.ModuleDict()
            first_block = None
            for i, block in enumerate(decoder_blocks):
                self.decoder_blocks.add_module(f"block_{i + 1}", block)
                if first_block is None:
                    first_block = block

        self.head = nn.Linear(num_channels, vocab_size, bias=False)
        if symmetric_embedding:
            self.head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)  # B,L,C

        x = x.permute(0, 2, 1)  # B,C,L

        for block in self.encoder_blocks.values():
            x = block(x)

        if self.decoder_blocks is not None:
            source = self.qk(x).permute(0, 2, 1)  # B,L,C
            q, k = torch.split(source, source.shape[-1] // 2, dim=-1)

            for block in self.decoder_blocks.values():
                x = block(x, query=q, key=k)

        x = x.permute(0, 2, 1)  # B,L,C

        logits = self.head(x)

        return logits


def create_conv_text_transformer(
        vocab_size: int,
        num_blocks: int,
        num_layers: int,
        num_channels: int,
        kernel_size: Union[int, Iterable[int]] = 7,
        dilation: Union[int, Iterable[int]] = 1,
        norm: Union[None, str, Type[nn.Module]] = None,
        activation: Union[None, str, Callable] = "gelu",
        residual: bool = True,
        with_decoder: bool = False,
        cheap: Union[bool, Iterable[bool]] = False,
        attention: Optional[str] = None,  # "QK", "QV", "KV", "QKV"
        attention_activation: Union[None, str, Callable] = "elu+1",
        diagonal_embedding: bool = True,
        symmetric_embedding: bool = True,
):
    return ConvTextTransformer(
        vocab_size=vocab_size,
        diagonal_embedding=diagonal_embedding,
        symmetric_embedding=symmetric_embedding,
        encoder_blocks=[
            ConvTextBlock(
                num_layers=num_layers,
                num_channels=num_channels,
                norm=norm if i < num_blocks - 1 else None,
                kernel_size=kernel_size,
                dilation=dilation,
                residual=residual,
                activation=activation,
                attention=attention,
                attention_activation=attention_activation,
                cheap=cheap,
            )
            for i in range(num_blocks)
        ],
        decoder_blocks=[
            ConvTextBlock(
                num_layers=num_layers,
                num_channels=num_channels,
                norm=norm if i < num_blocks - 1 else None,
                kernel_size=kernel_size,
                dilation=dilation,
                residual=residual,
                activation=activation,
                attention="QKin",
                attention_activation=attention_activation,
                cheap=cheap,
            )
            for i in range(num_blocks)
        ] if with_decoder else None,
    )



class TestConvTextTransformer(unittest.TestCase):

    def test_conv_text_transformer(self):
        vocab_size = 256
        for num_blocks, num_layers, num_channels, attention in (
                (1, 1, 16, None),
                (2, 3, 16, "QK"),
                (2, 3, 16, "QKV"),
        ):
            for cheap in (False, True):
                model = create_conv_text_transformer(
                    vocab_size=vocab_size,
                    num_blocks=num_blocks,

                    num_layers=num_layers,
                    num_channels=num_channels,
                    norm="bn1d",
                    attention=attention,
                    cheap=cheap,
                )
                inp = torch.randint(0, 255, (1, 100))

                try:
                    outp = model(inp)
                except:
                    print(model)
                    raise

                self.assertEqual(
                    torch.Size((1, 100, vocab_size)),
                    outp.shape,
                )
