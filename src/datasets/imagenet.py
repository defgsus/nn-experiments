import io
from pathlib import Path
from typing import Union, Optional, Iterable, Tuple, Literal

import PIL.Image
import pyarrow.parquet as pq
import torchvision.transforms.functional as VF

from src import config
from .base_iterable import BaseIterableDataset


class Imagenet1kIterableDataset(BaseIterableDataset):
    """
    first clone the huggingface repo https://huggingface.co/datasets/benjamin-paine/imagenet-1k

    Top-50 resolutions in 10% of imagenet:

        size        count   %
        (500, 371) 	623 	0.14
        (500, 348) 	646 	0.15
        (800, 533) 	660 	0.15
        (500, 343) 	661 	0.15
        (500, 361) 	665 	0.15
        (1600, 1200)667 	0.15
        (500, 344) 	695 	0.16
        (500, 342) 	697 	0.16
        (500, 340) 	697 	0.16
        (300, 225) 	697 	0.16
        (500, 350) 	706 	0.16
        (500, 341) 	711 	0.16
        (500, 378) 	723 	0.17
        (500, 373) 	731 	0.17
        (320, 240) 	748 	0.17
        (500, 356) 	782 	0.18
        (500, 368) 	791 	0.18
        (335, 500) 	795 	0.18
        (500, 337) 	797 	0.18
        (480, 360) 	824 	0.19
        (500, 281) 	830 	0.19
        (500, 339) 	861 	0.20
        (500, 358) 	920 	0.21
        (1024, 768) 968 	0.22
        (500, 377) 	970 	0.22
        (500, 338) 	1023 	0.24
        (500, 354) 	1052 	0.24
        (600, 400) 	1069 	0.25
        (500, 336) 	1090 	0.25
        (357, 500) 	1092 	0.25
        (500, 331) 	1188 	0.27
        (600, 450) 	1338 	0.31
        (332, 500) 	1385 	0.32
        (400, 500) 	1626 	0.37
        (400, 300) 	1626 	0.37
        (500, 376) 	1641 	0.38
        (800, 600) 	2384 	0.55
        (334, 500) 	2404 	0.55
        (500, 374) 	2572 	0.59
        (500, 357) 	3217 	0.74
        (640, 480) 	3372 	0.78
        (500, 335) 	3506 	0.81
        (500, 400) 	4385 	1.01
        (500, 500) 	4597 	1.06
        (500, 332) 	7112 	1.64
        (333, 500) 	10385 	2.39
        (500, 334) 	11121 	2.56
        (375, 500) 	17378 	4.00
        (500, 333) 	46627 	10.73
        (500, 375) 	99418 	22.88
    """
    _NUM_IMAGES = {
        "train": 1281167,
        "validation": 50000,
        "test": 100000,
    }

    def __init__(
            self,
            type: Literal["train", "validation", "test"] = "train",
            image_type: Literal["pil", "tensor"] = "tensor",
            with_label: bool = False,
            size_filter: Optional[Iterable[Tuple[int, int]]] = None,
            repo_path: Union[str, Path] = config.BIG_DATASETS_PATH / "hug" / "imagenet-1k",
    ):
        if type not in self._NUM_IMAGES:
            raise ValueError(f"'type' needs to be one of {', '.join(self._NUM_IMAGES)}, got '{type}'")
        super().__init__()
        self._type = type
        self._image_type = image_type
        self._with_label = with_label
        self._size_filter = set(size_filter) if size_filter is not None else None
        self._repo_path = Path(repo_path)

    def __len__(self):
        return self._NUM_IMAGES[self._type]

    def __iter__(self):
        files = sorted((self._repo_path / "data").glob(f"{self._type}-*-of-*.parquet"))
        for file in files:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=10):
                images = batch["image"]
                labels = batch["label"]

                for image, label in zip(images, labels):
                    buffer = io.BytesIO(image["bytes"].as_buffer())
                    image = PIL.Image.open(buffer)

                    if self._size_filter:
                        if image.size not in self._size_filter:
                            continue

                    if self._image_type == "tensor":
                        image = VF.to_tensor(image.convert("RGB"))
                    if self._with_label:
                        yield image, label
                    else:
                        yield image
