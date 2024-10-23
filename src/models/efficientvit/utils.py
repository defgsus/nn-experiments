import os
from inspect import signature
from typing import Optional, Union, Dict, Any, Callable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_kwargs_from_config(config: dict, target_func: Callable) -> Dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


def val2list(x: Union[list, tuple, Any], repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: Union[list, tuple, Any], min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def list_sum(x: list) -> Any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x: list) -> Any:
    return list_sum(x) / len(x)


def list_join(x: list, sep="\t", format_str="%s") -> str:
    return sep.join([format_str % val for val in x])


def get_device(model: nn.Module) -> torch.device:
    return model.parameters().__next__().device


def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def resize(
        x: torch.Tensor,
        size: Optional[Any] = None,
        scale_factor: Optional[List[float]] = None,
        mode: str = "bicubic",
        align_corners: Optional[bool] = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, is_distributed=True):
        self.is_distributed = is_distributed
        self.sum = 0
        self.count = 0

    def _sync(self, val: Union[torch.Tensor, int, float]) -> Union[torch.Tensor, int, float]:
        return sync_tensor(val, reduce="sum") if self.is_distributed else val

    def update(self, val: Union[torch.Tensor, int, float], delta_n=1):
        self.count += self._sync(delta_n)
        self.sum += self._sync(val * delta_n)

    def get_count(self) -> Union[torch.Tensor, int, float]:
        return self.count.item() if isinstance(self.count, torch.Tensor) and self.count.numel() == 1 else self.count

    @property
    def avg(self):
        avg = -1 if self.count == 0 else self.sum / self.count
        return avg.item() if isinstance(avg, torch.Tensor) and avg.numel() == 1 else avg


def dist_init() -> None:
    if is_dist_initialized():
        return
    try:
        torch.distributed.init_process_group(backend="nccl")
        assert torch.distributed.is_initialized()
    except Exception:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        print("warning: dist not init")


def is_dist_initialized() -> bool:
    return torch.distributed.is_initialized()


def get_dist_rank() -> int:
    return int(os.environ["RANK"])


def get_dist_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def is_master() -> bool:
    return get_dist_rank() == 0


def sync_tensor(tensor: Union[torch.Tensor, float], reduce: str = "mean") -> Union[torch.Tensor, List[torch.Tensor]]:
    if not is_dist_initialized():
        return tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.Tensor(1).fill_(tensor).cuda()
    tensor_list = [torch.empty_like(tensor) for _ in range(get_dist_size())]
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce == "mean":
        return list_mean(tensor_list)
    elif reduce == "sum":
        return list_sum(tensor_list)
    elif reduce == "cat":
        return torch.cat(tensor_list, dim=0)
    elif reduce == "root":
        return tensor_list[0]
    else:
        return tensor_list
