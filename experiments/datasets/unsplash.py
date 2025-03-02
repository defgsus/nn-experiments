from typing import Tuple

from src.datasets import UnsplashDataset
from src.util.image import set_image_channels


def unsplash_dataset(shape: Tuple[int, int, int], train: bool):
    if train:
        return (
            UnsplashDataset(data=("tensor",), max_size=160, cache_size=100)
            .transform([lambda i: set_image_channels(i, shape[0])])
            .skip(128)
            .repeat(3, per_item=True)
            .random_crop_all(shape[-2:])
            .to_iterable().shuffle(max_shuffle=100*3)
        )
    else:
        return (
            UnsplashDataset(data=("tensor",), max_size=160)
            .transform([lambda i: set_image_channels(i, shape[0])])
            .limit(128)
            .center_crop(shape[-2:])
            .freeze()
        )
