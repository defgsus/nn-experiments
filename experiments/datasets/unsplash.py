from typing import Tuple, Optional

from src.datasets import UnsplashDataset
from src.util.image import set_image_channels


def unsplash_dataset(shape: Optional[Tuple[int, int, int]], train: bool):
    if train:
        ds = (
            UnsplashDataset(data=("tensor",), max_size=160, cache_size=100)
            .skip(128)
        )
        if shape is not None:
            ds = ds.transform([lambda i: set_image_channels(i, shape[0])])

        ds = ds.repeat(3, per_item=True)
        if shape is not None:
            ds = ds.random_crop_all(shape[-2:])

        return ds.to_iterable().shuffle(max_shuffle=100*3)

    else:
        ds = (
            UnsplashDataset(data=("tensor",), max_size=160)
            .limit(128)
        )
        if shape is not None:
            ds = (
                ds
                .transform([lambda i: set_image_channels(i, shape[0])])
                .center_crop(shape[-2:])
            )

        return ds.freeze()
