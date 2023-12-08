"""
Based on example by Tony-Y
    at https://discuss.pytorch.org/t/differentiable-torch-histc/25865/4

"""
import torch


def soft_histogram_flat(x: torch.Tensor, bins: int, min: float, max: float, sigma: float = 100.):
    """
    Soft differentiable histogram.

    :param x: Tensor, input of any size, will be flattened
    :param bins: int, number of bins
    :param min: float, minimum value to consider
    :param max: float, maximum value to consider
    :param sigma: float, smoothing factor, higher is more prices. For high precision use something like 100_000.
    :return: Tensor of shape `(bins, )`
    """
    if x.ndim > 1:
        x = x.flatten(0)

    delta = (max - min) / bins
    centers = min + delta * (torch.arange(bins, device=x.device, dtype=x.dtype) + 0.5)

    x = torch.unsqueeze(x, 0) - torch.unsqueeze(centers, 1)
    x = torch.sigmoid(sigma * (x + delta / 2)) - torch.sigmoid(sigma * (x - delta / 2))
    x = x.sum(dim=-1)
    return x


def soft_histogram(x: torch.Tensor, bins: int, min: float, max: float, sigma: float = 100.):
    """
    Soft differentiable histogram on batches.

    :param x: Tensor, input of any size, will be flattened after 1st dimension
    :param bins: int, number of bins
    :param min: float, minimum value to consider
    :param max: float, maximum value to consider
    :param sigma: float, smoothing factor, higher is more prices. For high precision use something like 100_000.
    :return: Tensor of shape `(batch_size, bins, )`
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim == 2:
        pass
    else:
        x = x.flatten(1)

    delta = (max - min) / bins
    centers = min + delta * (torch.arange(bins, device=x.device, dtype=x.dtype) + 0.5)

    x = torch.unsqueeze(x, 1) - torch.unsqueeze(centers, 1)
    x = torch.sigmoid(sigma * (x + delta / 2)) - torch.sigmoid(sigma * (x - delta / 2))
    x = x.sum(dim=-1)
    return x

