from typing import Tuple

import torch
import torch.nn.functional as F


def pad(input: torch.Tensor, padding: Tuple[int, int], wrap: bool = False):
    """
    1d (batched) padding similar to `torch.nn.functional.pad`
    but supports `circular` padding with a padding size larger than the input size.
    """
    input_dim = input.ndim

    cut_it = False
    if wrap:

        # F.pad only supports 2D+ for "circular" mode
        if input.ndim == 1:
            input = input.unsqueeze(0)

        w = input.shape[-1]
        if padding[0] > w or padding[1] > w:
            num_rep = 1 + max(padding) // w
            input = input.repeat(*(1 for _ in input.shape[:-1]), num_rep)
            cut_it = True

    padded = F.pad(input, padding, mode="circular" if wrap else "constant")

    if cut_it:
        padded = padded[..., :w + sum(padding)]

    if input_dim == 1 and padded.ndim == 2:
        padded = padded.squeeze(0)

    return padded
