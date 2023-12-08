from typing import Optional, Tuple, Union

import torch


def differentiable_histogram(
        x: torch.Tensor,
        bins: Union[int, torch.Tensor],
        *,
        range: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Very naive implementation of a histogram:
        - it's slow
        - it is currently not really differentiable, the gradients get lost
        - otherwise it follows the `torch.histogram` method explicitly,
          unless bins is a non-incremental tensor
    """
    range_provided = range is not None
    if not range_provided:
        range = (float(x.min()), float(x.max()))
        if range[0] == range[1]:
            range = (range[0] - .5, range[0] + .5)

    if isinstance(bins, int):
        bin_table = torch.linspace(*range, bins + 1)
    else:
        if not isinstance(bins, torch.Tensor):
            raise TypeError(f"Expected `bins` to be of type int or Tensor, got {type(bins).__name__}")

        if bins.ndim != 1:
            raise ValueError(f"Expected `bins` to be one-dimensional, got {bins.shape}")

        if range_provided:
            raise TypeError(f"When `bins` is a Tensor, the `range` parameter must be unused, got {range}")

        bin_table = bins

    hist = torch.zeros(bin_table.shape[0] - 1).to(x.device)

    x = x.flatten(0)

    for dim, (bin_value, next_bin_value) in enumerate(zip(bin_table, bin_table[1:])):

        if dim + 2 < bin_table.shape[0]:
            mask = ((x >= bin_value) & (x < next_bin_value)).float()
        else:
            # for the last bin, include the final edge (<=)
            mask = ((x >= bin_value) & (x <= next_bin_value)).float()

        hist[dim] += ((x - x.detach()) + mask).sum()

    return hist, bin_table

