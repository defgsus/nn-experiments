from pathlib import Path
from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import torchvision.transforms.functional as VF

from PIL import Image, ImageFont, ImageDraw

from .base_iterable import BaseIterableDataset


class TextToPixelIterableDataset(BaseIterableDataset):

    def __init__(
            self,
            dataset: Union[Dataset, IterableDataset],
            screen_size: Tuple[int, int] = (24, 40),
            font_shape: Tuple[int, int] = (8, 8),
            font_file: Union[str, Path] = Path("~/.local/share/fonts/unscii-8.ttf").expanduser(),
    ):
        super().__init__()
        self._dataset = dataset
        self._screen_size = screen_size
        self._font_shape = font_shape
        self._font_map = {}
        self._font = ImageFont.truetype(str(font_file), min(self._font_shape))

    def __iter__(self):
        for item in self._dataset:
            is_tuple = isinstance(item, (tuple, list))
            if is_tuple:
                text = item[0]
                rest_args = item
            else:
                text = item
                rest_args = [item]

            image = self._render_text(text)

            yield image, *rest_args

    def _render_text(self, text: str):
        lines = text.splitlines()

        font_lines = []
        for y in range(self._screen_size[-2]):
            if y < len(lines):
                line = lines[y]
            else:
                font_lines.append(torch.zeros(1, self._screen_size[0] * self._font_shape[0], self._font_shape[1]))
                continue

            font_line = []
            for x in range(self._screen_size[-1]):
                if x < len(line):
                    ch = line[x]
                else:
                    ch = " "
                if ch not in self._font_map:
                    self._font_map[ch] = self._render_font(ch)

                font_line.append(self._font_map[ch])

            font_lines.append(torch.concat(font_line, dim=-1))

        return torch.concat(font_lines, dim=-2)

    def _render_font(self, ch: str):
        image = Image.new("L", (self._font_shape[1], self._font_shape[0]))
        draw = ImageDraw.ImageDraw(image)
        draw.text(
            (0, 0), ch,
            font=self._font,
            fill=(255,),
        )
        return VF.to_tensor(image)
