import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import lark

from src.models.util import activation_to_module
from src.models.util import ResidualAdd


_parser = None

def get_parser():
    global _parser

    if _parser is not None:
        return _parser

    grammar = r"""
    start: elements
    elements: element ("-" element)*
    element: layer | loop | default_assignment
    loop: UINT "x(" elements ")"
    layer: conv | activation_layer | batch_norm | residual | max_pool | avg_pool | dropout | fully_connected | global_max_pool | global_avg_pool
    
    activation_layer: ACTIVATION
    
    batch_norm: "bn"
    
    residual: "r(" elements ")"
    
    max_pool: "maxp" (kernel_size | stride | dilation | padding)*
    avg_pool: "avgp" (kernel_size | stride | padding)*
    
    global_max_pool: "gmaxp"
    global_avg_pool: "gavgp"
    
    dropout: "do" UFLOAT?
    
    fully_connected: "fc" UINT
    
    default_assignment: (kernel_size | stride | dilation | padding | activation)+
    
    conv: channels "x" (kernel_size | stride | dilation | padding | activation)*
    channels: UINT | UFLOAT
    kernel_size: "k" UINT | UINT
    stride: "s" UINT
    dilation: "d" UINT
    padding: "p" UINT
    activation: "a" ACTIVATION
    
    ACTIVATION: "relu" | "gelu" | "sigmoid" | "tanh"
    UINT: /0|[1-9]\d*/
    UFLOAT: UINT? "." /\d/*
    """
    class Transformer(lark.Transformer):
        def UINT(self, token: lark.Token):
            return int(token.value)
        def UFLOAT(self, token: lark.Token):
            return float(token.value)
        def ACTIVATION(self, token: lark.Token):
            return activation_to_module(token.value)

    _parser = lark.Lark(
        grammar,
        parser="lalr",
        transformer=Transformer(),
    )
    return _parser


class Context:
    def __init__(
            self,
            layers: nn.Sequential,
            input_shape: Tuple[int, int, int],
            default_conv_attrs: dict,
            previous_channels: int,
            previous_linear_channels: Optional[int] = None,
    ):
        self.input_shape = input_shape
        self.layers = layers
        self.default_conv_attrs = default_conv_attrs
        self.previous_channels = previous_channels
        self.previous_linear_channels = previous_linear_channels

    def copy(self):
        return self.__class__(
            input_shape=self.input_shape,
            layers=self.layers,
            default_conv_attrs=self.default_conv_attrs.copy(),
            previous_channels=self.previous_channels,
            previous_linear_channels=self.previous_linear_channels,
        )

    @torch.no_grad()
    def get_current_shape(self) -> Tuple[int, int, int]:
        inp = torch.zeros(1, *self.input_shape)
        outp = self.layers(inp)
        return outp.shape[1:]


def create_layers(
        script: str,
        input_shape: Tuple[int, int, int],
):
    parser = get_parser()
    tree = parser.parse(script)

    layers = nn.Sequential()
    context = Context(
        layers=layers,
        input_shape=input_shape,
        previous_channels=input_shape[0],
        default_conv_attrs={
            "kernel_size": 3,
            "stride": 1,
            "dilation": 1,
            "padding": 0,
            "bias": True,
        }
    )
    add_layers(context, tree)
    return context.layers


def add_layers(
        context: Context,
        tree: lark.Tree,
):
    if tree.data.value in ("start", "elements"):
        for ch in tree.children:
            add_layers(context, ch)

    elif tree.data.value == "element":
        if tree.children[0].data.value == "loop":
            for i in range(int(tree.children[0].children[0])):
                add_layers(context, tree.children[0].children[1])

        elif tree.children[0].data.value == "layer":
            add_layers(context, tree.children[0])

        elif tree.children[0].data.value == "default_assignment":
            for conv_attr in tree.children[0].children:
                context.default_conv_attrs[conv_attr.data.value] = conv_attr.children[0]

        else:
            raise ValueError(f"Invalid element content '{tree.children[0].data.value}'")

    elif tree.data.value == "layer":
        if tree.children[0].data.value == "conv":
            if context.previous_linear_channels is not None:
                raise ValueError(f"Can't add convolution after linear layer")

            conv_attrs = context.default_conv_attrs.copy()
            for conv_attr in tree.children[0].children:
                conv_attrs[conv_attr.data.value] = conv_attr.children[0]
            out_channels = conv_attrs.pop("channels")
            if isinstance(out_channels, float):
                out_channels = int(context.previous_channels * out_channels)
            act = conv_attrs.pop("activation", None)
            context.layers.append(
                nn.Conv2d(in_channels=context.previous_channels, out_channels=out_channels, **conv_attrs)
            )
            if act is not None:
                context.layers.append(act)
            context.previous_channels = out_channels

        elif tree.children[0].data.value == "activation_layer":
            act = tree.children[0].children[0]
            if act is not None:
                context.layers.append(act)

        elif tree.children[0].data.value == "batch_norm":
            if context.previous_linear_channels is None:
                context.layers.append(nn.BatchNorm2d(num_features=context.previous_channels))
            else:
                context.layers.append(nn.BatchNorm1d(num_features=context.previous_linear_channels))

        elif tree.children[0].data.value == "residual":
            sub_context = context.copy()
            sub_context.layers = nn.Sequential()
            add_layers(sub_context, tree.children[0].children[0])
            context.layers.append(ResidualAdd(sub_context.layers))
            context.previous_channels = sub_context.previous_channels
            context.previous_linear_channels = sub_context.previous_linear_channels

        elif tree.children[0].data.value in ("max_pool", "avg_pool"):
            if context.previous_linear_channels is not None:
                raise ValueError(f"Can't add pooling after linear layer")

            klass = nn.MaxPool2d if tree.children[0].data.value == "max_pool" else nn.AvgPool2d
            pool_attrs = {
                "kernel_size": 2,
                "stride": 1,
                "dilation": 1,
                "padding": 0,
            }
            for pool_attr in tree.children[0].children:
                pool_attrs[pool_attr.data.value] = pool_attr.children[0]
            if tree.children[0].data.value == "avg_pool":
                pool_attrs.pop("dilation")
            context.layers.append(klass(**pool_attrs))

        elif tree.children[0].data.value in ("global_max_pool", "global_avg_pool"):
            if context.previous_linear_channels is not None:
                raise ValueError(f"Can't add pooling after linear layer")

            channels, height, width = context.get_current_shape()
            klass = nn.MaxPool2d if tree.children[0].data.value == "global_max_pool" else nn.AvgPool2d
            context.layers.append(klass(kernel_size=(height, width)))
            context.layers.append(nn.Flatten(-3))
            context.previous_linear_channels = channels

        elif tree.children[0].data.value == "dropout":
            value = .5
            if tree.children[0].children:
                value = tree.children[0].children[0]
            if context.previous_linear_channels is None:
                context.layers.append(nn.Dropout2d(value))
            else:
                context.layers.append(nn.Dropout(value))

        elif tree.children[0].data.value == "fully_connected":
            channels_out = tree.children[0].children[0]
            if context.previous_linear_channels is None:
                channels_in = math.prod(context.get_current_shape())
                context.layers.append(nn.Flatten(-3))
            else:
                channels_in = context.previous_linear_channels
            context.layers.append(nn.Linear(channels_in, channels_out))
            context.previous_linear_channels = channels_out

        else:
            raise ValueError(f"Invalid layer content '{tree.children[0].data.value}'")

    else:
        raise ValueError(f"Invalid tree node '{tree.data.value}'")

