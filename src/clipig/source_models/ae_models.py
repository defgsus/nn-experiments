from .base import *


class AutoencoderModelHxW(SourceModelBase):

    NAME = "autoencoder_grid"
    IS_AUTOENCODER = True
    PARAMS = [
        *SourceModelBase.PARAMS,
        {
            "name": "autoencoder",
            "type": "autoencoder",
            #"default": "pixelart-3x32x32-128",
            "default": "pixelart-3x32x32-mlp-128-256",
        },
        {
            "name": "grid_size",
            "type": "int2",
            "default": [1, 1],
            "min": [1, 1],
        },
        {
            "name": "overlap",
            "type": "int2",
            "default": [0, 0],
            "min": [0, 0],
        },
    ]

    # note that `grid_size` and `overlap` are (x, y) tuples for UI convenience
    def __init__(
            self,
            autoencoder: nn.Module,
            autoencoder_shape: Tuple[int, int, int],
            code_size: int,
            grid_size: Tuple[int, int],
            overlap: Tuple[int, int] = (0, 0),
            std: float = .5,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.autoencoder_shape = autoencoder_shape
        self.code_size = code_size
        self.grid_size = tuple(reversed(grid_size))
        self.overlap = tuple(reversed(overlap))
        self.std = std
        self.code = nn.Parameter(torch.randn(math.prod(self.grid_size), code_size) * std)

    def forward(self):
        images = self.autoencoder.decoder(self.code).clamp(0, 1)

        s = images.shape
        gh, gw = self.grid_size[-2:]

        if self.overlap == (0, 0):
            output = torch.zeros_like(images).view(s[-3], s[-2] * gh, s[-1] * gw)
            for y in range(gh):
                for x in range(gw):
                    output[:, y * s[-2]: (y + 1) * s[-2], x * s[-1]: (x + 1) * s[-1]] = images[y * gw + x]

        else:
            output = torch.zeros(
                s[-3],
                s[-2] * gh - (gh - 1) * self.overlap[-2],
                s[-1] * gw - (gh - 1) * self.overlap[-1],
            ).to(images.device)
            output_sum = torch.zeros_like(output[0])
            window = get_image_window(s[-2:]).to(output.device)

            for y in range(gh):
                for x in range(gw):
                    yo = y * (s[-2] - self.overlap[-2])
                    xo = x * (s[-1] - self.overlap[-1])
                    output[:, yo: yo + s[-2], xo: xo + s[-1]] = output[:, yo: yo + s[-2], xo: xo + s[-1]] + images[y * gw + x] * window
                    output_sum[yo: yo + s[-2], xo: xo + s[-1]] = output_sum[yo: yo + s[-2], xo: xo + s[-1]] + window

            mask = output_sum > 0
            output[:, mask] = output[:, mask] / output_sum[mask]

        return output

    @torch.no_grad()
    def clear(self):
        self.code[:] = torch.zeros_like(self.code)

    @torch.no_grad()
    def randomize(self):
        with torch.no_grad():
            self.code[:] = torch.randn_like(self.code) * self.std

    @torch.no_grad()
    def set_image(self, image: torch.Tensor):
        if self.overlap == (0, 0):
            gh, gw = self.grid_size[-2:]
            th, tw = gh * self.autoencoder_shape[-2], gw * self.autoencoder_shape[-1]

            image = fit_image(image, shape=(self.autoencoder_shape[0], th, tw), dtype=self.code.dtype)

            patches = torch.concat([p.unsqueeze(0) for p in iter_image_patches(image, self.autoencoder_shape[-2:])])

        else:
            raise NotImplementedError("Sorry, not implemented yet")

        self.code[:] = self.autoencoder.encoder(patches)
