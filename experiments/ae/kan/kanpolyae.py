import math
from typing import Tuple, List, Union, Callable, Iterable

import torch
import torch.nn as nn

from src.models.kan import KANPolyLayer
from src.models.transform import Reshape
from src.util import param_make_tuple


class KANPolyAE(nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],
            channels: List[int],
            order: Union[int, Iterable[int]],
            encoder_activation: Union[None, str, Callable, Iterable[Union[None, str, Callable]]] = None,
            decoder_activation: Union[None, str, Callable, Iterable[Union[None, str, Callable]]] = None,
    ):
        super().__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        self.encoder.append(nn.Flatten(1))

        channels = list(channels)
        for i, (ch, next_ch, order_, enc_act, dec_act) in enumerate(zip(
                [math.prod(shape)] + channels,
                channels,
                param_make_tuple(order, len(channels), "order"),
                param_make_tuple(encoder_activation, len(channels), "encoder_activation"),
                list(reversed(param_make_tuple(decoder_activation, len(channels), "decoder_activation"))),
        )):
            is_last_encoder_layer = i == len(channels) - 1
            is_last_decoder_layer = i == 0

            self.encoder.append(KANPolyLayer(
                ch, next_ch, order_,
                activation=enc_act,
            ))
            self.decoder.insert(0, KANPolyLayer(
                next_ch, ch, order_,
                activation=dec_act,
            ))
        self.decoder.append(Reshape(shape))

    def forward(self, x):
        return self.decoder(self.encoder(x))
