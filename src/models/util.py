import itertools
import math
from typing import List, Iterable, Tuple, Optional, Callable, Union, Type, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF


def get_loss_callable(loss: Union[str, Callable, nn.Module]) -> Callable:
    if isinstance(loss, nn.Module) or callable(loss):
        return loss

    elif loss == "kl_div":
        return nn.KLDivLoss(reduction="batchmean")

    elif loss in ("l1", "mae"):
        return nn.L1Loss()

    elif loss in ("l2", "mse"):
        return nn.MSELoss()

    elif loss in ("bce", "binary_cross_entropy"):
        return F.binary_cross_entropy

    elif loss in ("ce", "cross_entropy"):
        return F.cross_entropy

    elif hasattr(F, loss) and callable(getattr(F, loss)):
        return getattr(F, loss)

    else:
        raise ValueError(f"Unexpected loss function '{loss}'")


class PrintLayer(nn.Module):
    def __init__(self, name: str = "debug"):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"{self.name}:", x.shape if isinstance(x, torch.Tensor) else type(x).__name__)
        return x


class ResidualAdd(nn.Module):

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


class ResidualConcat(nn.Module):

    def __init__(self, module: nn.Module, dim: int = 1):
        super().__init__()
        self.module = module
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.concat([x, self.module(x)], dim=self.dim)


class Lambda(nn.Module):
    def __init__(self, func: Callable):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def extra_repr(self):
        return f"func={self.func}"


def activation_to_module(
        activation: Union[None, str, Callable, nn.Module, Type[nn.Module]]
) -> Union[None, nn.Module]:
    if activation is None:
        return None

    if isinstance(activation, nn.Module):
        return activation

    try:
        if issubclass(activation, nn.Module):
            return activation()
    except TypeError:
        pass

    if callable(activation):
        return Lambda(activation)

    if isinstance(activation, str):
        s = activation.lower()
        if s == "none":
            return None
        for module in (torch.nn, ):
            for key, value in vars(module).items():
                try:
                    if key.lower() == s and issubclass(value, nn.Module):
                        return value()
                except TypeError:
                    pass

    raise ValueError(f"Unrecognized activation: {repr(activation)}")


def activation_to_callable(
        activation: Union[None, str, Callable, nn.Module, Type[nn.Module]]
) -> Optional[Callable]:
    if activation is None:
        return None

    if isinstance(activation, nn.Module):
        return activation

    try:
        if issubclass(activation, nn.Module):
            return activation()
    except TypeError:
        pass

    if callable(activation):
        return activation

    if isinstance(activation, str):
        s = activation.lower()
        if s == "none":
            return None
        # catch `torch.tanh` before `nn.tanh`
        for module in (torch, F, torch.nn):
            for key, value in vars(module).items():
                if key.lower() == s and callable(value):
                    try:
                        if issubclass(value, nn.Module):
                            return value()
                    except TypeError:
                        pass
                    return value

    raise ValueError(f"Unrecognized activation: {repr(activation)}")


def normalization_to_module(
        normalization: Union[None, str, Type[nn.Module]],
        *args,
        channels: Optional[int] = None,
        **kwargs,
) -> Optional[Callable]:
    if normalization is None:
        return None

    if isinstance(normalization, nn.Module):
        return normalization

    try:
        if issubclass(normalization, nn.Module):
            return normalization(*args, **kwargs)
    except TypeError:
        pass

    if isinstance(normalization, str):
        s = normalization.lower()
        if s == "rms":
            from src.models.mamba.mamba import RMSNorm
            return RMSNorm(d_model=channels, **kwargs)
        elif s in ("bn1d", "batchnorm1d"):
            return nn.BatchNorm1d(num_features=channels, **kwargs)
        elif s in ("bn2d", "batchnorm2d"):
            return nn.BatchNorm2d(num_features=channels, **kwargs)
        for module in (torch.nn, ):
            for key, value in vars(module).items():
                try:
                    if key.lower() == s and issubclass(value, nn.Module):
                        return value(*args, **kwargs)
                except TypeError as e:
                    pass

    raise ValueError(f"Unrecognized normalization: {repr(normalization)}")


@torch.no_grad()
def get_model_weight_images(
        model: nn.Module,
        grad_only: bool = True,
        max_channels: int = 16,
        min_size: int = 2,
        max_size: int = 128,
        normalize: str = "all",  # "each", "shape", "all", "none"
        size_to_scale: Dict[int, float] = {10: 4, 20: 2},
):
    from torchvision.utils import make_grid
    from src.util.image import signed_to_image

    # yield 2d shapes
    def _iter_params():
        for param in model.parameters():
            if not param.requires_grad and grad_only:
                continue
            if param.ndim == 2:
                yield param
            elif param.ndim == 3:
                for ch in range(min(max_channels, param.shape[0])):
                    yield param[ch]
            elif param.ndim == 4:
                for ch in range(min(max_channels, param.shape[0])):
                    yield param[ch, 0]
                for ch in range(min(max_channels, param.shape[1])):
                    yield param[0, ch]

    shape_dict = {}
    for param in _iter_params():
        if any(s < min_size for s in param.shape):
            continue
        param = param[:max_size, :max_size]

        scale = None
        for key in sorted(size_to_scale):
            value = size_to_scale[key]
            if all(s <= key for s in param.shape):
                scale = value
                break

        if scale:
            param = VF.resize(
                param.unsqueeze(0),
                [s * scale for s in param.shape], VF.InterpolationMode.NEAREST, antialias=False
            ).squeeze(0)

        if param.shape not in shape_dict:
            shape_dict[param.shape] = []
        shape_dict[param.shape].append(param)

    grids = []
    for shape in sorted(shape_dict):
        params = shape_dict[shape]
        nrow = max(1, int(math.sqrt(len(params)) * 2))
        if normalize == "each":
            grids.append(make_grid([signed_to_image(p) for p in params], nrow=nrow))
        else:
            grids.append(make_grid([p.unsqueeze(0) for p in params], nrow=nrow))

    max_width = max(g.shape[-1] for g in grids)

    for image_idx, image in enumerate(grids):
        if image.shape[-1] < max_width:
            grids[image_idx] = VF.pad(image, [0, 0, max_width - image.shape[-1], 0])

    if normalize == "shape":
        grids = [signed_to_image(g) for g in grids]

    grids = torch.concat([
        VF.pad(grid, [0, 0, 0, 2])
        for grid in grids
    ], dim=-2)

    if normalize == "all":
        grids = signed_to_image(grids)

    return grids
