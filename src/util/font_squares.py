from pathlib import Path
from typing import Union, Tuple, Optional

import torch
import torchvision.transforms.functional as VF
import PIL.Image, PIL.ImageFont, PIL.ImageDraw


class FontSquares:
    def __init__(
            self,
            file: Union[str, Path] = Path("~/.local/share/fonts/unscii-8.ttf").expanduser(),
            shape: Tuple[int, int, int] = (1, 8, 8),
            center: bool = False,
    ):
        """
        Generator for monospaced images from text

        :param file: file to use as font
        :param shape: tuple of (C, H, W), square channels & size, C must be 1 or 3
        :param center: bool, center all fonts into their squares
        """
        assert len(shape) == 3, f"Expected [C, H, W] shape, got {shape}"
        assert shape[0] in (1, 3), f"Expected 1 or 3 channels, got {shape}"

        self.shape = shape
        self.center = center
        self.font = PIL.ImageFont.truetype(
            str(file),
            size=min(self.shape[-2:]),
        )
        self._font_map = {}

    def __call__(self, ch: Union[str, int, torch.Tensor], dim: int = 2) -> torch.Tensor:
        """
        Convert text to image

        :param ch: int (ordinal) or str (single character or text)
        :param dim: int, dimension for concatenation/stacking
        :return: Tensor of shape
            [C, H, W] if single character
            [C, H, W * N] if dim == 2
            [C, H * N, N] if dim == 1
            [C * N, H, W] if dim == 0
            [N, C, H, W] if dim == -1
        """
        if isinstance(ch, torch.Tensor):
            assert ch.ndim in (0, 1), f"Expect 0 or 1 dimension, got {ch.shape}"
            if ch.shape == torch.Size() or ch.shape[0] == 1:
                ch = ch.item()
            else:
                squares = [self(c.item()) for c in ch]
                if dim == -1:
                    return torch.stack(squares)
                else:
                    return torch.cat(squares, dim)

        elif isinstance(ch, str):
            if len(ch) == 0:
                ch = 32
            elif len(ch) == 1:
                ch = ord(ch)
            else:
                squares = [self(c) for c in ch]
                if dim == -1:
                    return torch.stack(squares)
                else:
                    return torch.cat(squares, dim)

        ch = max(ch, 32) if ch != 0 else 0

        if ch not in self._font_map:
            if ch == 0:
                self._font_map[ch] = torch.ones(self.shape)
            else:
                image = PIL.Image.new(
                    "RGB" if self.shape[0] == 3 else "L",
                    (self.shape[-1], self.shape[-2]),
                )
                draw = PIL.ImageDraw.ImageDraw(image)

                if self.center:
                    L, T, R, B = draw.textbbox((0, 0), chr(ch), font=self.font)
                    xy = (
                        (self.shape[-1] - (R - L)) // 2,
                        -T + (self.shape[-2] - (B - T)) // 2,
                    )
                else:
                    xy = (0, 0)

                draw.text(
                    xy,
                    chr(ch),
                    font=self.font,
                    fill=(255,) * self.shape[0],
                )
                self._font_map[ch] = VF.to_tensor(image)

        return self._font_map[ch]

    def reverse(self, image: torch.Tensor, dim: int = 2) -> str:
        """
        Convert image back to text by best-match.

        :param image: Tensor of shape
            [C, H, W * N] if dim == 2
            [C, H * N, N] if dim == 1
            [C * N, H, W] if dim == 0
            [N, C, H, W] if dim == -1
        :param dim: int, dimension where image is concatenated/stacked
        :return: str
        """
        if dim == -1:
            assert image.ndim == 4, f"Expected 4 dimensions, got {image.shape}"
            assert image.shape[1:] == self.shape, f"Expected square shape of {self.shape}, got {image.shape}"
            squares = image
        elif dim == 0:
            assert image.ndim == 3, f"Expected 3 dimensions, got {image.shape}"
            assert image.shape[0] % self.shape[0] == 0, f"Expected channels divisible by {self.shape[0]}, got {image.shape}"
            assert image.shape[1] == self.shape[1], f"Expected height of {self.shape[1]}, got {image.shape}"
            assert image.shape[2] == self.shape[2], f"Expected width of {self.shape[2]}, got {image.shape}"
        elif dim == 1:
            assert image.ndim == 3, f"Expected 3 dimensions, got {image.shape}"
            assert image.shape[0] == self.shape[0], f"Expected {self.shape[0]} channels, got {image.shape}"
            assert image.shape[1] % self.shape[1] == 0, f"Expected height divisible by {self.shape[1]}, got {image.shape}"
            assert image.shape[2] == self.shape[2], f"Expected width of {self.shape[2]}, got {image.shape}"
        elif dim == 2:
            assert image.ndim == 3, f"Expected 3 dimensions, got {image.shape}"
            assert image.shape[0] == self.shape[0], f"Expected {self.shape[0]} channels, got {image.shape}"
            assert image.shape[1] == self.shape[1], f"Expected height of {self.shape[1]}, got {image.shape}"
            assert image.shape[2] % self.shape[2] == 0, f"Expected width divisible by {self.shape[2]}, got {image.shape}"
        else:
            raise NotImplementedError(f"Expected dim in -1, 0, 1 or 2, got {dim}")

        if dim >= 0:
            squares = image.split(self.shape[dim], dim)

        all_ords = list(self._font_map)
        all_fonts = torch.cat([self(o).unsqueeze(0) for o in all_ords]).to(image)
        output = []
        for square in squares:
            diffs = (all_fonts - square).abs().flatten(1).mean(1)
            best = all_ords[diffs.argmin()]
            # print(chr(best), best)
            output.append(chr(best))
        return "".join(output)
