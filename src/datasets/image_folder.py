import math
from pathlib import Path
import glob
import warnings
import random
from typing import Union, Generator, Optional, Callable, Any, Dict, List, Tuple

import PIL.Image

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torchvision.datasets import ImageFolder as TorchImageFolder, DatasetFolder
from torchvision.datasets.folder import is_image_file
from torchvision.transforms.functional import pil_to_tensor

from src.util.image import image_resize_crop, set_image_channels, set_image_dtype
from .base_iterable import BaseIterableDataset


class ImageFolderIterableDataset(BaseIterableDataset):

    def __init__(
            self,
            root: Union[str, Path],
            recursive: bool = False,
            shuffle: bool = False,
            max_images: Optional[int] = None,
            max_bytes: Optional[int] = None,
            force_channels: Optional[int] = None,
            force_dtype: Optional[torch.dtype] = torch.float32,
            with_filename: bool = False,
            verbose: bool = False,
    ):
        super().__init__()
        self.root = Path(root).expanduser()
        self.recursive = recursive
        self.max_images = max_images
        self.max_bytes = max_bytes
        self.force_channels = force_channels
        self.force_dtype = force_dtype
        self.with_filename = with_filename
        self.shuffle = shuffle
        self.verbose = verbose
        self._filenames = None

    def __len__(self):
        """
        This is only an approximation,
        files that can not be loaded by PIL will be skipped
        """
        self._get_filenames()
        size = len(self._filenames)
        if self.max_images is not None:
            size = min(size, self.max_images)
        return size

    # def __getitem__(self, index) -> Union[None, PIL.Image.Image, torch.Tensor]:
    #     """
    #     files that can not be loaded by PIL will None
    #     """
    #     self._get_filenames()
    #     filename = self._filenames[index]
    #     return self._load_filename(filename)

    def __iter__(self) -> Generator[Union[torch.Tensor, Tuple[torch.Tensor, str]], None, None]:
        self._get_filenames()

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            filenames = self._filenames
        else:
            filenames = self._filenames[worker_info.id::worker_info.num_workers]

        if self.shuffle:
            filenames = filenames.copy()
            random.shuffle(filenames)

        # print("YIELDING", len(filenames), worker_info)
        count = 0
        count_bytes = 0
        for filename in filenames:
            image = self._load_filename(filename)
            if image is not None:

                count += 1
                count_bytes += math.prod(image.shape) * 4
                if self.verbose:
                    shape_str = "x".join(str(s) for s in image.shape)
                    print(f"{self.__class__.__name__}: images={count:,}, bytes={count_bytes:,}, image={shape_str} {filename}")

                if self.with_filename:
                    yield image, filename
                else:
                    yield image

                if self.max_images is not None and count >= self.max_images:
                    if self.verbose:
                        print(f"{self.__class__.__name__}: break because num images {count:,} >= {self.max_images:,}")
                    break

                if self.max_bytes is not None and count_bytes >= self.max_bytes:
                    if self.verbose:
                        print(f"{self.__class__.__name__}: break because num bytes {count_bytes:,} >= {self.max_bytes:,}")
                    break

    def _load_filename(self, filename: str) -> Union[None, PIL.Image.Image, torch.Tensor]:
        try:
            image = PIL.Image.open(filename)
        except (PIL.Image.DecompressionBombError, PIL.UnidentifiedImageError) as e:
            warnings.warn(f"Error reading image '{filename}': {type(e).__name__}: {e}")
            return None

        image = pil_to_tensor(image)

        if self.force_dtype is not None:
            image = set_image_dtype(image, self.force_dtype)

        if self.force_channels is not None:
            image = set_image_channels(image, self.force_channels)

        return image

    def _get_filenames(self):
        if self._filenames is None:

            if self.root.is_file():
                self._filenames = [str(self.root)]

            else:
                glob_path = self.root
                if self.recursive:
                    glob_path /= "**/*"
                else:
                    glob_path /= "*"

                self._filenames = []
                for filename in glob.glob(str(glob_path), recursive=self.recursive):
                    if is_image_file(filename):
                        self._filenames.append(filename)

                self._filenames.sort()

