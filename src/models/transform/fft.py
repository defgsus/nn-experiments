from typing import Tuple

import torch
import torch.nn as nn


class FFTLayer(nn.Module):
    """
    Converts an n-dim input to fourier space.

    if `allow_complex==False`, the output shape for images (B, C, H, W) will be:

        type   concat_dim  output shape
        fft    -1          B, C, H, W * 2
        rfft   -1          B, C, H, W + 2
        hfft   -1          B, C, H, W * 2 - 2

        fft    -2          B, C, H * 2, W
        rfft   -2          B, C, H * 2, W // 2 + 1
        hfft   -2          B, C, H, W * 2 - 2        # hfft does not produce complex data so `concat_dim` is unused

        fft    -2          B, C * 2, H, W
        rfft   -2          B, C * 2, H, W // 2 + 1
        hfft   -2          B, C, H, W * 2 - 2

    if `allow_complex==True`, the output might be complex data and shapes are:

        type   output shape          is complex
        fft    B, C, H, W            yes
        rfft   B, C, H, W // 2 + 1   yes
        hfft   B, C, H, W * 2 - 2    no
    """
    def __init__(
            self,
            type: str = "fft",
            allow_complex: bool = False,
            concat_dim: int = -1,
            norm: str = "forward",
            inverse: bool = False,
    ):
        super().__init__()
        supported_types = [
            name[:-1] for name in dir(torch.fft)
            if name.endswith("fftn") and not name.startswith("i")
        ]
        if type not in supported_types:
            raise ValueError(f"Expected `type` to be one of {', '.join(supported_types)}, got '{type}'")

        supported_norm = ("forward", "backward", "ortho")
        if norm not in supported_norm:
            raise ValueError(f"Expected `norm` to be one of {', '.join(supported_norm)}, got '{norm}'")

        self.type = type
        self.norm = norm
        self.allow_complex = allow_complex
        self.concat_dim = concat_dim
        self.inverse = inverse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inverse and not self.allow_complex and not torch.is_complex(x) and not self.type == "hfft":
            x = torch.complex(
                    torch.slice_copy(x, self.concat_dim, 0, x.shape[self.concat_dim] // 2),
                    torch.slice_copy(x, self.concat_dim, x.shape[self.concat_dim] // 2),
            )

        func_name = f"{'i' if self.inverse else ''}{self.type}n"
        output = getattr(torch.fft, func_name)(x, norm=self.norm)

        if not self.inverse:
            if not self.allow_complex and torch.is_complex(output):
                output = torch.concat([output.real, output.imag], dim=self.concat_dim)
        else:
            output = output.real

        return output
