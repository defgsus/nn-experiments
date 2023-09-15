from typing import Union

import torch


def to_torch_device(device: Union[None, str, torch.device], auto_cuda: bool = True) -> torch.device:
    if device is None or device == "auto":
        device = torch.device("cuda" if auto_cuda and torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    return device
