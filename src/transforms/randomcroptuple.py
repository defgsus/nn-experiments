import torch
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms.functional as VF
from torchvision.transforms import RandomCrop
from src.datasets.base_dataset import BaseDataset, IterableDataset


class RandomCropTuple(RandomCrop):
    """
    Same as torchvision's RandomCrop but applies the same crop
    to all same-sized images in a list or tuple.
    """
    def forward(self, img_or_batch):
        if not isinstance(img_or_batch, (list, tuple)):
            return super().forward(img_or_batch)

        new_values = list(img_or_batch)
        size = None
        crop_args = None
        for idx, img in enumerate(new_values):
            if isinstance(img, torch.Tensor):
                if size is None or VF.get_dimensions(img)[-2:] == size:
                    size = VF.get_dimensions(img)[-2:]

                    if self.padding is not None:
                        img = VF.pad(img, self.padding, self.fill, self.padding_mode)

                    height, width = size
                    # pad the width if needed
                    if self.pad_if_needed and width < self.size[1]:
                        padding = [self.size[1] - width, 0]
                        img = VF.pad(img, padding, self.fill, self.padding_mode)
                    # pad the height if needed
                    if self.pad_if_needed and height < self.size[0]:
                        padding = [0, self.size[0] - height]
                        img = VF.pad(img, padding, self.fill, self.padding_mode)

                    if crop_args is None:
                        crop_args = self.get_params(img, self.size)

                    new_values[idx] = VF.crop(img, *crop_args)

        return new_values if isinstance(img_or_batch, list) else tuple(new_values)


