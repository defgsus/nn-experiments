from typing import Union

import torch


def mix(a: torch.Tensor, b: torch.Tensor, x: Union[float, torch.Tensor]):
    if isinstance(x, torch.Tensor):
        while x.ndim < a.ndim:
            x = x[..., None]

    return a * (1. - x) + x * b

def smooth(x):
    return torch.pow(x, 2) * (3. - 2. * x)

def smooth2(x):
    return torch.pow(x, 3) * (x * (x * 6. - 15.) + 10.)

def step(x, mi, ma):
    return torch.clip((x - mi) / (ma - mi), 0, 1)

def smooth_step(x, mi, ma):
    return smooth(step(x, mi, ma))

def smooth2_step(x, mi, ma):
    return smooth2(step(x, mi, ma))
