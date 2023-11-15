from typing import Iterable, Union

import torch
import torch.nn as nn


class DecoderEnsemble(nn.Module):

    def __init__(
            self,
            *decoders: nn.Module,
            weights: Union[None, Iterable[float], torch.Tensor] = None,
            train_weights: bool = True,
    ):
        super().__init__()
        self.decoders = nn.ModuleDict({
            f"decoder_{i + 1}": decoder
            for i, decoder in enumerate(decoders)
        })

        if weights is None:
            weights = torch.ones(len(self.decoders)) / len(self.decoders)
        elif isinstance(weights, torch.Tensor):
            pass
        else:
            weights = torch.Tensor(weights)
        self.weights = nn.Parameter(weights, requires_grad=train_weights)

    def forward(self, *args, **kwargs):
        output_sum = None
        for i, decoder in enumerate(self.decoders.values()):
            output = decoder(*args, **kwargs) * self.weights[i]

            if output_sum is None:
                output_sum = output
            else:
                output_sum = output_sum + output

        return output_sum
