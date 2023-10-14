from typing import Tuple, Optional, Iterable, Union, List

import torch
import torchvision.transforms.functional as VF
import PIL.Image
import PIL.ImageDraw


@torch.no_grad()
def plot_audio(
        waves: Union[torch.Tensor, List[torch.Tensor]],
        shape: Union[int, Tuple[int, int]] = 128,
        colors: Optional[Iterable[Tuple[int, int, int]]] = None,
        tensor: bool = False,
        rate: int = 44_100,
) -> Union[PIL.Image.Image, torch.Tensor]:

    if isinstance(waves, (tuple, list)):
        waves = torch.concat([w.unsqueeze(0) for w in waves])

    if waves.ndim == 1:
        waves = waves.unsqueeze(0)

    if colors is None:
        colors = _DEFAULT_COLORS
    else:
        colors = tuple(colors)

    length = waves.shape[-1]

    if isinstance(shape, int):
        shape = (shape, int(shape * length / rate))
    else:
        shape = tuple(shape)

    image = PIL.Image.new("RGB", (shape[-1], shape[-2]))
    draw = PIL.ImageDraw.ImageDraw(image)

    draw.line(
        ((0, shape[-2] / 2), (shape[-1], shape[-2] / 2)),
        fill=(128, 128, 128),
        width=1,
    )

    for i, wave in enumerate(waves):
        x = torch.linspace(0, shape[-1] - 1, len(wave)).to(wave.device)
        y = (1. - wave) * (shape[-2] / 2. - 1)
        segments = torch.cat([x.unsqueeze(0), y.unsqueeze(0)]).permute(1, 0)
        segments = tuple(tuple(s) for s in segments)
        draw.line(segments, width=1, fill=colors[i % len(colors)])

    if tensor:
        return VF.to_tensor(image)
    return image


_DEFAULT_COLORS = (
    (194, 194, 194),
    (128, 255, 128),
    (255, 128, 128),
    (128, 128, 255),
    (255, 255, 128),
    (255, 128, 255),
    (128, 255, 255),
    ( 32, 255,  32),
    (255,  32,  32),
    ( 32,  32, 255),
    (255, 255,  32),
    (255,  32, 255),
    ( 32, 255, 255),
)
