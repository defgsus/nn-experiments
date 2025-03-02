import json
import zipfile
from pathlib import Path
from typing import Union, Optional, Tuple, Dict

import pandas as pd
import PIL.Image
import torch
import torchvision.transforms.functional as VF

from src import config
from src.datasets import BaseDataset
from src.util import streaming_download


class UnsplashDataset(BaseDataset):

    _WEB_URL = "https://unsplash.com/data/lite/latest"
    _ZIP_FILENAME = "latest.zip"

    # see issue https://github.com/unsplash/datasets/issues/61
    _URL_FIX = {
        "rsJtMXn3p_c": "https://images.unsplash.com/9/vectorbeastcom-grass-sun.jpg",
        "vigsqYux_-8": "https://images.unsplash.com/reserve/vof4H8A1S02iWcK6mSAd_sarahmachtsachen.com_TheBeach.jpg",
        "9_9hzZVjV8s": "https://images.unsplash.com/reserve/RFDKkrvXSHqBaWMMl4W5_Heavy_company",
    }
    _DROP_IDS = [
        "NvL3xrjEX3k", "dkmjbTrHUEo", "Bvoc14Xyt00", "gEb091vDXOs",
        "Srlss1BXN1k", "GFttLy00kEE", "pduv-eZsfeI", "uIrfFEDXcLE",
        "vJkk9ELe-wk", "PnrrSnzyU8s", "MmWy_lqDQNE", "lIWy4xWEtKo",
        "Yi-bJNtHziA", "gEbJwcbWNpc", "_mu0vTle47w", "Ah3ir6fxfqg",
        "iXBGCA5AhZQ", "3t78mz5oVx8", "uhyiwekQbsM", "EdLtou2WSxU",
        "QvbLjB9pHLM",
    ]

    def __init__(
            self,
            data: Tuple[str] = ("tensor", "data"),
            base_dir: Union[str, Path] = config.BIG_DATASETS_PATH / "unsplash",
            max_size: Optional[int] = None,
            cache_size: int = 10,
    ):
        """
        The Unsplash Lite Dataset: https://github.com/unsplash/datasets/

        Accessing an item of this dataset will first download the compressed zip-file
        and then download each image separately.

        :param data: tuple of str, define the data that is emitted by this dataset. Can be
            a combination of these items: "tensor", "pil", "data"
        :param base_dir: Path | str, the folder in which the dataset is stored
        :param max_size: int, optional.
            Limits the largest edge of the images. Setting a value will download all images again!
            If unset, the images will be stored in original resolution as jpeg, which takes
            many hours to complete.
            If set, the images will be scaled by the unsplash backend and downloaded/stored as png
            to a different sub-folder.

            In any case, downloading the images takes a couple of hours.
            Check the `download_unsplash` function below to increase download speed for smaller
            images with multiple threads.
        """
        self._data = data
        self._base_dir = Path(base_dir)
        self._max_size = max_size
        self._table: Optional[pd.DataFrame] = None
        self._cache_size = cache_size
        self._cache: Dict[Path, dict] = {}
        self._cache_time: int = 0
        self._cache_misses: int = 0

    def _download(self) -> Path:
        zip_filename = self._base_dir / self._ZIP_FILENAME
        if not zip_filename.exists():
            streaming_download(
                url=self._WEB_URL,
                local_filename=zip_filename,
                verbose=True,
            )
        return zip_filename

    def get_table(self) -> pd.DataFrame:
        if self._table is None:
            with zipfile.ZipFile(self._download()) as zf:
                with zf.open("photos.tsv000") as fp:
                    self._table = (
                        pd.read_csv(fp, delimiter="\t")
                        .set_index("photo_id")
                        .drop(self._DROP_IDS, axis=0)
                    )
            for key, url in self._URL_FIX.items():
                if key in self._table:
                    self._table[key]["photo_image_url"] = url

            self._table.reset_index(inplace=True)

        return self._table

    def __len__(self):
        return self.get_table().shape[0]

    def __getitem__(self, index: int):
        row = self.get_table().iloc[index]

        folder_name = "images"
        format = "jpeg"
        params = None
        if self._max_size is not None:
            folder_name = f"images-maxs{self._max_size}"
            format = "png"
            w, h = row["photo_width"], row["photo_height"]
            if h > self._max_size and h > w:
                params = {"h": self._max_size, "fm": format}
            elif w > self._max_size and w >= h:
                params = {"w": self._max_size, "fm": format}

        photo_id = row["photo_id"]
        filename = self._base_dir / folder_name / f"{photo_id}.{format}"
        if not filename.exists():
            try:
                streaming_download(
                    row["photo_image_url"], filename, params=params,
                    timeout=10,
                )
            except Exception as e:
                raise IOError(f"{type(e).__name__}: {e} for image:\n{json.dumps(row.to_dict(), indent=2)}")

        return_data = []

        for what in self._data:
            if what == "tensor":
                return_data.append(self._get_tensor_image(filename))

            elif what == "pil":
                return_data.append(self._get_pil_image(filename))

            elif what == "data":
                return_data.append(row.to_dict())

            else:
                raise ValueError(f"Invalid data type '{what}'")

        if not return_data:
            return
        elif len(return_data) == 1:
            return return_data[0]
        else:
            return tuple(return_data)

    def _get_pil_image(self, filename):
        if filename not in self._cache:
            self._cache_misses += 1
            image = PIL.Image.open(filename)
            if "pil" in self._data:
                self._store_cache(filename, "pil", image)
            return image
        cache = self._cache[filename]
        cache["time"] = self._cache_time
        self._cache_time += 1
        return cache["pil"]

    def _get_tensor_image(self, filename):
        if filename not in self._cache:
            self._cache_misses += 1
            image = PIL.Image.open(filename)
            if "pil" in self._data:
                self._store_cache(filename, "pil", image)
            tensor_image = VF.to_tensor(image)
            self._store_cache(filename, "tensor", tensor_image)
            return tensor_image
        cache = self._cache[filename]
        cache["time"] = self._cache_time
        self._cache_time += 1
        return cache["tensor"]

    def _store_cache(self, filename: Path, type: str, image: Union[PIL.Image.Image, torch.Tensor]):
        if filename not in self._cache:
            self._cache[filename] = {
                "time": self._cache_time
            }
            if len(self._cache) > self._cache_size:
                oldest = sorted(self._cache.keys(), key=lambda k: self._cache[k]["time"])[0]
                if oldest != filename:
                    del self._cache[oldest]

        cache = self._cache[filename]
        cache[type] = image


def download_unsplash(ds: UnsplashDataset, workers: int = 4):
    from tqdm import tqdm
    from multiprocessing.pool import ThreadPool

    indices = list(range(len(ds)))

    with tqdm(indices) as progress:

        def _get_item(index: int):
            _ = ds[index]
            progress.update()

        pool = ThreadPool(workers)
        pool.map(_get_item, indices)


if __name__ == "__main__":
    # This averages at about 1.2mb/s download traffic
    # takes about 4 hours and downloads 9.2 GB
    download_unsplash(UnsplashDataset(data=tuple(), max_size=640), workers=4)
