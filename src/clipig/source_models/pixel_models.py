from .base import *


class PixelModel(SourceModelBase):

    NAME = "pixels"
    PARAMS = [
        *SourceModelBase.PARAMS,
        {
            "name": "channels",
            "type": "select",
            "default": "RGB",
            "choices": ["L", "RGB"]
        },
        {
            "name": "size",
            "type": "int2",
            "default": [224, 224],
            "min": [1, 1],
            "max": [4096, 4096],
        },
    ]

    def __init__(
            self,
            size: Tuple[int, int],  # x,y for UI convenience
            channels: str,
    ):
        super().__init__()
        channel_map = {"L": 1, "RGB": 3, "HSV": 3}
        num_channels = channel_map.get(channels, 3)
        self.shape = (num_channels, size[1], size[0])
        self.code = nn.Parameter(torch.empty(self.shape))

    def forward(self):
        return self.code

    @torch.no_grad()
    def clear(self):
        self.code[:] = torch.zeros_like(self.code)

    @torch.no_grad()
    def randomize(self, mean: float, std: float):
        self.code[:] = (torch.randn_like(self.code) * std + mean).clamp(0, 1)

    @torch.no_grad()
    def set_image(self, image: torch.Tensor):
        image = fit_image(image, self.shape, self.code.dtype)
        self.code[:] = image


class PixelHSVModel(PixelModel):

    NAME = "pixels_hsv"
    PARAMS = [
        *SourceModelBase.PARAMS,
        {
            "name": "channels",
            "type": "select",
            "default": "HSV",
            "choices": ["L", "HSV"]
        },
        {
            "name": "size",
            "type": "int2",
            "default": [224, 224],
            "min": [1, 1],
            "max": [4096, 4096],
        },
    ]

    def forward(self):
        return hsv_to_rgb(set_image_channels(super().forward(), 3))

    @torch.no_grad()
    def randomize(self, mean: float, std: float):
        if self.shape[0] == 3:
            self.code[:1] = torch.rand_like(self.code[:1])
            self.code[1:] = (torch.randn_like(self.code[1:]) * std + mean).clamp(0, 1)
        else:
            self.code[:] = (torch.randn_like(self.code) * std + mean).clamp(0, 1)

    @torch.no_grad()
    def set_image(self, image: torch.Tensor):
        super().set_image(rgb_to_hsv(set_image_channels(image, 3)))
