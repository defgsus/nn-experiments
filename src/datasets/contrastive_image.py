from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset
import torchvision.transforms as VT
import torchvision.transforms.functional as VF


class ContrastiveImageDataset(Dataset):
    """
    Returns tuple of two image crops and bool if the crops
    are from the same image.
    """
    def __init__(
            self,
            source_dataset: Dataset,
            crop_shape: Tuple[int, int],
            num_crops: int = 2,
            num_contrastive_crops: int = 2,
            prob_h_flip: float = .5,
            prob_v_flip: float = .5,
            prob_hue: float = .5,
            prob_saturation: float = .5,
            prob_brightness: float = .5,
            prob_grayscale: float = 0.,
            generator: Optional[torch.Generator] = None
    ):
        self.source_dataset = source_dataset
        self.crop_shape = crop_shape
        self.num_contrastive_crops = num_contrastive_crops
        self.num_crops = num_crops
        self.prob_h_flip = prob_h_flip
        self.prob_v_flip = prob_v_flip
        self.prob_hue = prob_hue
        self.prob_saturation = prob_saturation
        self.prob_brightness = prob_brightness
        self.prob_grayscale = prob_grayscale
        self.generator = torch.Generator() if generator is None else generator

        transforms = [self._crop]
        if prob_h_flip:
            transforms.append(self._h_flip)
        if prob_v_flip:
            transforms.append(self._v_flip)
        if prob_hue:
            transforms.append(self._hue)
        if prob_saturation:
            transforms.append(self._saturation)
        if prob_brightness:
            transforms.append(self._brightness)
        if prob_grayscale:
            transforms.append(self._to_grayscale)
        self.cropper = VT.Compose(transforms)

    def __len__(self) -> int:
        return len(self.source_dataset) * (self.num_crops + self.num_contrastive_crops)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        all_crops = self.num_crops + self.num_contrastive_crops

        true_index = index // all_crops
        crop_index = index % all_crops

        image1 = image2 = self._get_image(true_index)
        is_same = True

        if crop_index >= self.num_crops:
            other_index = true_index
            while other_index == true_index:
                other_index = torch.randint(0, len(self.source_dataset) - 1, (1,), generator=self.generator).item()

            image2 = self._get_image(other_index)
            is_same = False

        return (
            self.cropper(image1),
            self.cropper(image2),
            is_same,
        )

    def _get_image(self, index: int):
        image = self.source_dataset[index]
        if isinstance(image, (tuple, list)):
            image = image[0]
        return image

    def _crop(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        x = torch.randint(0, h - self.crop_shape[0] + 1, size=(1,), generator=self.generator).item()
        y = torch.randint(0, w - self.crop_shape[1] + 1, size=(1,), generator=self.generator).item()

        return VF.crop(image, y, x, self.crop_shape[0], self.crop_shape[1])

    def _h_flip(self, image: torch.Tensor) -> torch.Tensor:
        doit = torch.rand(1, generator=self.generator).item() < self.prob_h_flip
        return VF.hflip(image) if doit else image

    def _v_flip(self, image: torch.Tensor) -> torch.Tensor:
        doit = torch.rand(1, generator=self.generator).item() < self.prob_v_flip
        return VF.vflip(image) if doit else image

    def _hue(self, image: torch.Tensor) -> torch.Tensor:
        amt = torch.rand(1, generator=self.generator).item() - .5
        return VF.adjust_hue(image, amt)

    def _saturation(self, image: torch.Tensor) -> torch.Tensor:
        amt = torch.rand(1, generator=self.generator).item() * 2.
        return VF.adjust_saturation(image, amt)

    def _brightness(self, image: torch.Tensor) -> torch.Tensor:
        amt = torch.rand(1, generator=self.generator).item() + .5
        return VF.adjust_brightness(image, amt)

    def _to_grayscale(self, image: torch.Tensor) -> torch.Tensor:
        doit = torch.rand(1, generator=self.generator).item() < self.prob_grayscale
        return VF.rgb_to_grayscale(image, image.shape[0]) if doit else image
