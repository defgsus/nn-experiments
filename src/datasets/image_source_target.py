import random
from pathlib import Path
from typing import Optional, Dict, Union, Tuple

import PIL.Image
import torch
import torchvision.transforms.functional as VF

from src.datasets.base_dataset import BaseDataset
from torchvision.datasets.folder import is_image_file


class ImageSourceTargetDataset(BaseDataset):
    def __init__(
            self,
            path: Union[str, Path],
            source_subpath: str = "source",
            target_subpath: str = "target",
            adversarials_subpath: str = "adversarial",
            target_first: bool = False,
            adversarial_rate: float = 0.,
    ):
        path = Path(path)
        self._source_path = path / source_subpath
        self._target_path = path / target_subpath
        self._adversarials_path = path / adversarials_subpath
        self._source_images: Dict[str, Optional[torch.Tensor]] = {}
        self._target_images: Dict[str, Optional[torch.Tensor]] = {}
        self._adversarials_images: Dict[str, Optional[torch.Tensor]] = {}
        self._target_first = target_first
        self._adversarial_rate = adversarial_rate

        for filename in sorted(self._source_path.glob("*")):
            if is_image_file(str(filename)):
                self._source_images[filename.name] = VF.to_tensor(PIL.Image.open(filename))

        for filename in sorted(self._target_path.glob("*")):
            if is_image_file(str(filename)):
                self._target_images[filename.name] = VF.to_tensor(PIL.Image.open(filename))

        if self._adversarial_rate > 0:
            for filename in sorted(self._adversarials_path.glob("*")):
                if is_image_file(str(filename)):
                    self._adversarials_images[filename.name] = VF.to_tensor(PIL.Image.open(filename))

        if sorted(self._source_images) != sorted(self._target_images):
            raise RuntimeError(f"Source and target filenames are not identical")

        print(f"datatset: {len(self._source_images)} images, {len(self._adversarials_images)} adversarials")

        self._index = {
            i: key
            for i, key in enumerate(self._source_images)
        }

    def __len__(self):
        return len(self._source_images)

    def __getitem__(self, idx: int):
        key = self._index[idx]

        target, source = self._target_images[key], self._source_images[key]

        if self._adversarial_rate > 0 and key in self._adversarials_images:
            if random.uniform(0, 1) < self._adversarial_rate:
                source = self._adversarials_images[key]

        if self._target_first:
            return target, source
        else:
            return source, target


class ImageSourceTargetCropDataset(BaseDataset):
    def __init__(
            self,
            path: Union[str, Path],
            shape: Tuple[int, int],
            num_crops: int,  # per image
            source_subpath: str = "source",
            target_subpath: str = "target",
            target_first: bool = False,
            random: bool = False,
            adversarial_rate: float = 0.,
    ):
        self._dataset = ImageSourceTargetDataset(
            path=path, source_subpath=source_subpath, target_subpath=target_subpath, target_first=target_first,
            adversarial_rate=adversarial_rate,
        )
        self._shape = shape
        self._num_crops = num_crops
        self._random = random
        if not self._random:
            self._crop_positions = []
            rng = globals()["random"].Random(23)
            for idx in range(len(self._dataset) * self._num_crops):
                image_idx = idx % len(self._dataset)
                source_image, target_image = self._dataset[image_idx]
                self._crop_positions.append((image_idx, *self._get_crop_pos(source_image, rng)))

    def __len__(self):
        return len(self._dataset) * self._num_crops

    def __getitem__(self, idx: int):
        if self._random:
            image_idx = random.randrange(len(self._dataset))

            source_image, target_image = self._dataset[image_idx]
            assert source_image.shape == target_image.shape

            x, y = self._get_crop_pos(source_image, random)
        else:
            image_idx, x, y = self._crop_positions[idx]
            source_image, target_image = self._dataset[image_idx]
            assert source_image.shape == target_image.shape

        return (
            source_image[..., y: y + self._shape[0], x: x + self._shape[1]],
            target_image[..., y: y + self._shape[0], x: x + self._shape[1]],
        )

    def _get_crop_pos(self, image: torch.Tensor, rng: random.Random) -> Tuple[int, int]:
        H, W = image.shape[-2:]
        if self._shape[0] > H or self._shape[1] > W:
            raise RuntimeError(f"Crop shape {self._shape} is too large for image {image.shape}")
        x = rng.randrange(W - self._shape[1])
        y = rng.randrange(H - self._shape[0])
        return x, y
