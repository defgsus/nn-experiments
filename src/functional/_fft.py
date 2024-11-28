import torch


def normalized_fft2(
        image: torch.Tensor,
        exponent: float = 1.,
):
    size = image.shape[-1] * image.shape[-2]

    y = torch.fft.fft2(image)

    if exponent == 1.:
        y_r = y.real / size
        y_i = y.imag / size
    else:
        y_r = (y.real / size).abs().pow(1. / exponent) * y.real.sign()
        y_i = (y.imag / size).abs().pow(1. / exponent) * y.imag.sign()

    return torch.cat([y_r, y_i], dim=-3)


def normalized_ifft2(
        y: torch.Tensor,
        exponent: float = 1.,
):
    assert y.shape[-3] % 2 == 0, f"Expected channels dividable by 2, got {y.shape}"

    chan = y.shape[-3] // 2
    size = y.shape[-1] * y.shape[-2]

    if exponent == 1.:
        y = y * size
        y = torch.complex(
            y[..., :chan, :, :],
            y[..., chan:, :, :],
        )
    else:
        y_sign = y.sign()
        y = y.abs().pow(exponent) * size
        y = torch.complex(
            y[..., :chan, :, :] * y_sign[..., :chan, :, :],
            y[..., chan:, :, :] * y_sign[..., chan:, :, :],
            )

    image = torch.fft.ifft2(y).real

    return image
