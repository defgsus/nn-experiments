import math
from io import BytesIO
from typing import Union, Dict

import PIL.Image
import torch
import torchvision.transforms.functional as VF


class ImageCompressionRatio:
    COMPRESSION_FORMAT_OPTIONS = {
        "jpeg": {
            "low": {"quality": 100},
            "high": {"quality": 0},
        },
        "png": {
            "none": {"optimize": False, "compress_level": 0},
            "low": {"optimize": False, "compress_level": 1},
            "high": {"optimize": True, "compress_level": 9},
        }
    }

    def __init__(self):
        self._compression_reference = {}

    def __call__(
            self,
            image: Union[PIL.Image.Image, torch.Tensor],
            format: str = "png",
            quality: str = "high",
    ) -> float:
        """
        Calc ratio of compressed size / compressed size of black image
        """
        options = self.COMPRESSION_FORMAT_OPTIONS[format]

        if hasattr(image, "shape"):
            shape = image.shape
        else:
            shape = (len(image.mode), *image.size)

        reference_key = (shape, format)
        if reference_key not in self._compression_reference:
            if options.get("none"):
                reference_value = self._compressed_size(
                    image, format, options["none"],
                )
            else:
                reference_value = math.prod(shape)

            self._compression_reference[reference_key] = reference_value

        compressed_size = self._compressed_size(image, format, options[quality])
        return compressed_size / self._compression_reference[reference_key]

    def all(
            self,
            image: Union[PIL.Image.Image, torch.Tensor],
            prefix: str = "",
            suffix: str = "",
    ) -> Dict[str, float]:
        result = {}
        for format, qualities in self.COMPRESSION_FORMAT_OPTIONS.items():
            for key in qualities:
                if key != "none":
                    result[f"{prefix}{format}-{key}{suffix}"] = self(image, format=format, quality=key)
        return result

    def _compressed_size(
            self,
            image: Union[PIL.Image.Image, torch.Tensor],
            format: str,
            options: dict,
    ) -> int:
        if not isinstance(image, PIL.Image.Image):
            image = VF.to_pil_image(image)
        fp = BytesIO()
        image.save(fp, format, **options)
        return fp.tell()
