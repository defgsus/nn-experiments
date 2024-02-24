import json
import math
import os
import zipfile
import fnmatch
import sys
import tempfile
from pathlib import Path
from functools import partial
from typing import List, Iterable, Tuple, Optional, Callable, Union

from tqdm import tqdm
import PIL.Image
from PIL import ImageFont, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
import torchaudio.transforms as AT
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset, ConcatDataset
from torchvision.utils import make_grid
import requests

from src.datasets import *
from src.util.image import *
from ..base_iterable import BaseIterableDataset


class RpgTileIterableDataset(BaseIterableDataset):

    WEB_HOST = "https://opengameart.org"
    DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent / "tile-config.ndjson"

    def __init__(
            self,
            shape: Tuple[int, int, int] = (3, 32, 32),
            tile_config: Union[str, Path, List[dict]] = DEFAULT_CONFIG_FILE,
            directory: Union[str, Path] = Path(tempfile.gettempdir()) / "datasets" / "rpg_tile_dataset",
            include: Optional[str] = None,
            exclude: Optional[str] = None,
            even_files: Optional[bool] = None,
            interleave: bool = False,
            with_name: bool = False,
            with_pos: bool = False,
            download: bool = True,
            verbose: bool = False,
    ):
        self.shape = shape
        self.directory = Path(directory)
        self.interleave = interleave
        self.with_name = with_name
        self.with_pos = with_pos
        self.download = download
        self.verbose = verbose

        if isinstance(tile_config, (tuple, list)):
            self.tile_config = list(tile_config)
        else:
            with open(tile_config) as fp:
                self.tile_config = [
                    json.loads(line)
                    for line in fp.readlines()
                ]

        for i, entry in enumerate(self.tile_config):
            if entry.get("ignore_tiles"):
                entry["ignore_tiles"] = [tuple(t) for t in entry["ignore_tiles"]]

        if even_files is True:
            self.tile_config = self.tile_config[::2]
        elif even_files is False:
            self.tile_config = self.tile_config[1::2]

        if include is not None:
            self.tile_config = list(filter(
                lambda t: fnmatch.fnmatch(t["name"], include),
                self.tile_config
            ))
        if exclude is not None:
            self.tile_config = list(filter(
                lambda t: not fnmatch.fnmatch(t["name"], exclude),
                self.tile_config
            ))

    def __iter__(self):
        if not self.interleave:
            for params in self.tile_config:
                yield from self._iter_tiles(**params)
        else:
            iterables = [
                self._iter_tiles(**params)
                for params in self.tile_config
            ]
            while iterables:
                next_iterables = []
                for it in iterables:
                    try:
                        yield next(it)
                        next_iterables.append(it)
                    except StopIteration:
                        pass
                iterables = next_iterables

    def _iter_tiles(
            self,
            name: str,
            url: str,
            shape: Tuple[int, int],
            offset: Tuple[int, int] = None,
            stride: Optional[Tuple[int, int]] = None,
            limit: Optional[Tuple[int, int]] = None,
            remove_transparent: bool = True,
            ignore_lines: Iterable[int] = None,
            ignore_tiles: Iterable[Tuple[int, int]] = None,
    ):
        self._get_file(name, url)

        if ignore_lines:
            ignore_lines = set(ignore_lines)
        if ignore_tiles:
            ignore_tiles = set(ignore_tiles)

        image = PIL.Image.open(
            (Path(self.directory) / name).expanduser()
        )
        if image.mode == "P":
            image = image.convert("RGBA")
        image = VF.to_tensor(image)

        if image.shape[0] != self.shape[0]:
            if image.shape[0] == 4 and remove_transparent:
                image = image[:3] * image[3].unsqueeze(0)

            image = set_image_channels(image[:3], self.shape[0])

        if limit:
            image = image[..., :limit[0], :limit[1]]
        if offset:
            image = image[..., offset[0]:, offset[1]:]

        for patch, pos in iter_image_patches(image, shape, stride=stride, with_pos=True):
            pos = tuple(int(p) // s for p, s in zip(pos, shape))

            if ignore_lines and pos[0] in ignore_lines:
                continue
            if ignore_tiles and pos in ignore_tiles:
                continue

            if patch.std(1).mean() > 0.:
                patch = VF.resize(patch, self.shape[-2:], VF.InterpolationMode.NEAREST, antialias=False)

                if not self.with_pos and not self.with_name:
                    yield patch
                else:
                    patch = [patch]
                    if self.with_name:
                        patch.append(name)
                    if self.with_pos:
                        patch.append(pos)

                    yield tuple(patch)

    def _get_file(self, name: str, url: str):
        filename = self.directory / name
        if filename.exists():
            return

        name_parts = name.split("/")
        page_name = name_parts[0]
        page_directory = self.directory / page_name

        #if not (page_directory / "index.html").exists():
        #    os.makedirs(page_directory, exist_ok=True)
        #    response = requests.get(f"{self.WEB_HOST}/{page_name}")
        #    assert response.status_code == 200, f"Got status {response.status_code} from {response.request.url}"
        #    (page_directory / "index.html").write_text(response.text)

        url_name = url.split("/")[-1]
        download_name = (page_directory / url_name)

        if not download_name.exists():
            if self.verbose:
                print(f"downloading {url}", file=sys.stderr)
            response = requests.get(url)
            assert response.status_code == 200, f"Got status {response.status_code} from {response.request.url}"
            os.makedirs(page_directory, exist_ok=True)
            download_name.write_bytes(response.content)

        if not url.lower().endswith(".zip"):
            return

        sub_name = "/".join(name_parts[2:])

        with zipfile.ZipFile(download_name) as zipf:
            for file in zipf.filelist:
                if not file.is_dir():
                    if file.filename == sub_name:
                        fp = zipf.open(file.filename)
                        sub_filename = Path(str(download_name)[:-4]) / file.filename
                        if not sub_filename.exists():
                            if self.verbose:
                                print(f"extracting {file.filename} -> {sub_filename}", file=sys.stderr)
                            os.makedirs(sub_filename.parent, exist_ok=True)
                            sub_filename.write_bytes(fp.read())
                            return

        raise RuntimeError(
            f"File {name} not found in {download_name}"
        )
