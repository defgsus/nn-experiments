import math
import random
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable

from tqdm import tqdm
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torch.utils.data import DataLoader

from src.datasets import ImageFolderIterableDataset, ImageAugmentation, IterableShuffle, TotalCADataset


def main(
        shape=(3, 32, 32),
        dtype=torch.float32,
        image_folder="~/Pictures/__diverse/",
        output_filename="diverse-32x32.pt",
        max_megabyte=1_024,
):

    def create_augmentation():
        return torch.nn.Sequential(
            VT.RandomAffine(degrees=20, scale=(.3, 4), translate=(.5, .5)),
            VT.RandomPerspective(p=.5, distortion_scale=.7),
            VT.RandomInvert(p=.3),
            VT.RandomVerticalFlip(),
            VT.RandomHorizontalFlip(),
        )

    def augmented_dataset(ds, num_aug=1):
        return ImageAugmentation(
            ds,
            augmentations=[
                create_augmentation()
                for i in range(num_aug)
            ],
            final_shape=shape[-2:],
            final_channels=shape[0],
            final_dtype=dtype,
        )

    ds = ImageFolderIterableDataset(
        root=Path(image_folder).expanduser(),
        #root=Path("~/Pictures/eisenach/").expanduser(),
    )
    ds = augmented_dataset(ds, num_aug=32)
    #ds = IterableShuffle(ds, max_shuffle=5)

    tensor_batch = []
    tensor_size = 0
    last_print_size = 0
    for image in tqdm(ds):
        if isinstance(image, (tuple, list)):
            image = image[0]

        if len(image.shape) < 4:
            image = image.unsqueeze(0)

        tensor_batch.append(image)
        tensor_size += math.prod(image.shape) * 4

        if tensor_size - last_print_size > 1024 * 1024:
            last_print_size = tensor_size

            print(f"size: {tensor_size:,}")

        if tensor_size >= max_megabyte * 1024 * 1024:
            break

    tensor_batch = torch.cat(tensor_batch)
    torch.save(tensor_batch, output_filename)


if __name__ == "__main__":
    main()
