import math
from typing import Optional, IO, Union, Tuple, Callable

import torch


class UC:
    """
    Unicode shortcuts
    """
    FULL_BLOCK = "█"


class CC:
    """
    Console Colors
    """

    Off = "\x1b[0m"

    @classmethod
    def rgb(cls, r: Union[int, float], g: Union[int, float], b: Union[int, float]) -> str:
        r = max(0, min(255, r if isinstance(r, int) else int(r * 255)))
        g = max(0, min(255, g if isinstance(g, int) else int(g * 255)))
        b = max(0, min(255, b if isinstance(b, int) else int(b * 255)))
        return f"\x1b[38;2;{r};{g};{b}m"

    @classmethod
    def palette_signed(cls, v: float) -> str:
        v = max(-1., min(1., v))
        return cls.rgb(
            r=math.pow(max(0., v), .5),
            g=abs(v),
            b=math.pow(max(0., -v), .5),
        )

    @classmethod
    def palette_gray(cls, v: float) -> str:
        v = max(0., min(1., v))
        return cls.rgb(v, v, v)


def print_tensor_2d(
        tensor: torch.Tensor,
        signed: Optional[bool] = None,
        file: Optional[IO[str]] = None,
):
    assert tensor.ndim == 2, tensor.ndim

    with torch.no_grad():
        if signed is None:
            signed = torch.any(tensor < 0)
        tensor, (t_min, t_max), normalizer = _normalize_tensor(tensor, signed=signed)

        if signed:
            palette = lambda v: CC.palette_signed(v)
        else:
            palette = lambda v: CC.palette_gray(v)

        for row in tensor:
            print(
                "".join(
                    f"{palette(v)}{UC.FULL_BLOCK}"
                    for v in row
                ) + CC.Off,
                file=file,
            )
        print(
            f"range: "
            f"{palette(normalizer(t_min))}{t_min:.5f}{CC.Off} - "
            f"{palette(normalizer(t_max))}{t_max:.5f}{CC.Off}",
            file=file,
        )


def _normalize_tensor(
        tensor: torch.Tensor, signed: bool
) -> Tuple[
        torch.Tensor,
        Tuple[float, float],
        Callable[[Union[float, torch.Tensor]], Union[float, torch.Tensor]],
    ]:

    t_min, t_max = tensor.min().float(), tensor.max().float()
    if t_min == t_max:
        if t_max:
            return tensor / t_max, (t_min, t_max), lambda x: x / t_max

        return tensor, (t_min, t_max), lambda x: x

    if not signed:
        return (
            (tensor - t_min) / (t_max - t_min),
            (t_min, t_max),
            lambda x: (x - t_min) / (t_max - t_min)
        )

    return (
        tensor / max(abs(t_min), abs(t_max)),
        (t_min, t_max),
        lambda x: x / max(abs(t_min), abs(t_max))
    )
