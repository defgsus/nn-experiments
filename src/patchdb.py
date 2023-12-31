import gzip
import json
import os
import io
import sys
from pathlib import Path
import base64
import warnings
from typing import List, Iterable, Tuple, Optional, Callable, Union, Dict, Generator

import torch
import torchvision.transforms.functional as VF
import numpy as np
import faiss
import PIL.Image
from tqdm import tqdm

from src.util.image import *
from src.models.encoder import Encoder2d


class PatchDB:

    def __init__(
            self,
            filename: Union[str, Path],
            writeable: bool = False,
            max_image_cache: int = 100,
            verbose: bool = False,
            offset: Optional[int] = None,
            limit: Optional[int] = None,
            patch_shape: Optional[Tuple[int, int, int]] = None,
            interpolation: VF.InterpolationMode = VF.InterpolationMode.BILINEAR,
            encoder: Optional[Encoder2d] = None,
    ):
        """
        Constructs a database instance around `filename`

        PatchDB stores filenames, patch rectangles and corresponding feature embeddings.

        :param filename: str/Path, usually something like `database.patchdb`.
            If the file does not exists, PatchDB must be writeable!
        :param writeable: bool,
            If False, no changes are done to the database, but the file must exists.
            If True, `clear()` and `add_patch()` can be used.
        :param max_image_cache: int,
            Maximum number of images simultaneously in cache.
        :param verbose: bool
            If True, show progress during loading
        :param offset: optional int, skip first `offset` entries when reading
        :param limit: optional int, read at most `limit` entries
        :param patch_shape, optional CxHxW shape of desired patches,
            returned by `PatchDBIndex.Patch.patch`. Default is None or the `encoder.shape´.
        :param encoder: optional encoder to use for image encoding,
            should match the encoder used for creating the embeddings
        :param interpolation: torchvision interpolation mode used when resizing patches to `patch_shape`.
        """
        self.filename = Path(filename)
        self.max_image_cache = int(max_image_cache)
        self.verbose = bool(verbose)
        self._writeable = bool(writeable)
        self._offset = offset
        self._limit = limit
        if patch_shape is not None:
            self.patch_shape = tuple(patch_shape)
        else:
            self.patch_shape = None if encoder is None else encoder.shape

        self.interpolation = interpolation
        self.encoder = encoder

        self._fp = None
        self._image_cache: Dict[Path, Dict] = {}
        self._cache_counter: int = 0

        if not self._writeable and not self.filename.exists():
            raise RuntimeError(f"non-writeable PatchDB requested, but file does not exists: {self.filename}")

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        Close the database file (if open).

        Once the database is closed, all read operations work properly.
        It's still possible to add further patches to the database.
        """
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def clear(self):
        """
        Careful! Deletes the database file!
        """
        if not self._writeable:
            return

        self.close()
        if self.filename.exists():
            os.remove(self.filename)
            if self.verbose:
                print(f"PathDB deleted: {self.filename}", file=sys.stderr)

    def size_bytes(self) -> int:
        """Returns current size of database file"""
        if self.filename.exists():
            return self.filename.stat().st_size
        return 0

    def flush(self):
        """
        Flush the database file.
        It's still not readable, though, before `close` is called.
        """
        if self._fp:
            self._fp.flush()

    def add_patch(
            self,
            filename: Union[str, Path],
            rect: Union[Tuple[int, int, int, int], Iterable[int]],
            embedding: Optional[Iterable[float]] = None,
    ):
        """
        Add a patch to the database.

        No duplicate check!

        :param filename: str/Path, the image filename
        :param rect: sequence of ints: (top, left, height, width)
            The rectangle of the image patch matching the embedding.
            The values can be directly passed to `torchvision.transforms.function.crop`.
        :param embedding:
            A 1-dimensional vector. All embeddings must have the same length!
            Can be None if `PatchDB.encoder` is defined, in which case the image is loaded
            and the patch is cropped and encoded.
        :return:
        """
        rect = [int(r) for r in rect]

        if not self._writeable:
            warnings.warn("PatchDB.add_patch called on non-writeable db")
            return

        if embedding is None:
            if self.encoder is None:
                raise RuntimeError(f"Must specify `embedding` when no encoder is defined")
            image = self.image(filename)
            patch = VF.crop(image, *rect)
            embedding = self.encoder.encode_image(patch.unsqueeze(0))[0]

        embedding = self._to_numpy(embedding)
        
        filename = Path(filename).resolve()
        self._write({
            "filename": str(filename),
            "rect": rect,
            "embedding": base64.b64encode(embedding.data).decode("ascii"),
        })

    def iter_patches(self, desc: Optional[str] = None) -> Generator[Dict, None, None]:
        """
        Yields all stored patches (limited by `offset` and `limit`).

        Only use this on closed database!

        Use `PatchDB.index()` for loading all patches to memory.

        :param desc: optional str for `tqdm` progress bar, if `verbose` is True.
        :return: Generator of dict
        """
        iterable = self._read_lines()
        if self.verbose:
            iterable = tqdm(iterable, desc=desc or "reading PatchDB")
        for data in iterable:
            yield {
                **data,
                "embedding": np.frombuffer(base64.b64decode(data["embedding"]), dtype=np.float32),
            }

    def index(self) -> "PatchDBIndex":
        """
        Load all necessary data to memory.

        :return: `PatchDBIndex` instance
        """
        return PatchDBIndex(self)

    def image(self, filename: Union[str, Path]) -> torch.Tensor:
        """
        Load an image file and convert to Tensor.

        Internally, `max_image_cache` recently required images are cached.

        :param filename: str/Path, the filename of the image.
            Used unmodified as cache key.
        :return: Tensor
        """
        convert_to_1_color = False
        if self.patch_shape is not None and self.patch_shape[0] == 1:
            convert_to_1_color = True

        filename = Path(filename)
        self._cache_counter += 1
        if filename not in self._image_cache:

            img = VF.to_tensor(PIL.Image.open(filename))
            if convert_to_1_color:
                img = set_image_channels(img, 1)

            self._image_cache[filename] = {
                "age": self._cache_counter,
                "image": img,
            }
            if len(self._image_cache) > self.max_image_cache:
                oldest = sorted(self._image_cache, key=lambda fn: self._image_cache[fn]["age"])[0]
                del self._image_cache[oldest]

        img = self._image_cache[filename]
        img["age"] = self._cache_counter
        return img["image"]

    def to_new_patchdb(
            self,
            filename: Union[str, Path],
            include_filenames: Optional[Iterable[str]] = None,
            exclude_filenames: Optional[Iterable[str]] = None,
            offset: int = 0,
            limit: Optional[int] = None,
            encoder: Optional[Encoder2d] = None,
            transforms: Optional[Iterable[Callable]] = None
    ):
        """
        Creates a new database from the existing one.

        :param filename: filename of new database.
            If it exists, data will be appended.
        :param include_filenames: optional set of image filenames to filter for.
        :param exclude_filenames: optional set of image filenames to exclude.
        :param offset: int, skip first N entries
        :param limit: int, limit size of new database
        :param encoder: optional Encoder to re-create the embeddings
        :param transforms: optional list of callables that transform an image patch
            before it is passed to the encoder.
            Wither `encoder` or `PatchDB.encoder` must be defined for this.

        :return: new PatchDB instance
        """
        if include_filenames is not None:
            include_filenames = set(include_filenames)
        if exclude_filenames is not None:
            exclude_filenames = set(exclude_filenames)

        if transforms is not None:
            transforms = list(transforms)

        def _iter_patches(batch_size: int = 100):
            count = 0 if offset is None else -offset
            patch_batch, image_patch_batch, embedding_batch = [], [], []
            for patch in self.iter_patches(desc="copying database"):

                if include_filenames is not None and patch["filename"] not in include_filenames:
                    continue
                if exclude_filenames is not None and patch["filename"] in exclude_filenames:
                    continue

                count = count + 1

                if count < 0:
                    continue

                if limit is not None and count >= limit:
                    break

                if encoder is not None or transforms is not None:
                    image = self.image(patch["filename"])
                    image_patch = VF.crop(image, *patch["rect"])
                    image_patch = VF.resize(image_patch, self.patch_shape[-2:])

                    if transforms is not None:
                        for tr in transforms:
                            image_patch = tr(image_patch)

                    image_patch_batch.append(image_patch.unsqueeze(0))
                else:
                    embedding_batch.append(patch["embedding"])

                patch_batch.append({
                    "filename": patch["filename"],
                    "rect": patch["rect"],
                })

                if len(patch_batch) >= batch_size:
                    if image_patch_batch:
                        assert encoder is not None or self.encoder is not None, \
                            "Encoder must be present when transforming patches"

                        embedding_batch = encoder.encode_image(torch.concat(image_patch_batch))
                        image_patch_batch.clear()

                    yield patch_batch, embedding_batch
                    patch_batch.clear()
                    embedding_batch = []

            if len(patch_batch):
                if image_patch_batch:
                    embedding_batch = encoder.encode_image(torch.concat(image_patch_batch))

                yield patch_batch, embedding_batch

        db = PatchDB(
            filename=filename,
            writeable=True,
            verbose=self.verbose,
            patch_shape=self.patch_shape,
            interpolation=self.interpolation,
            encoder=self.encoder if encoder is None else encoder,
        )
        with db:
            for patch_batch, embedding_batch in _iter_patches():
                for patch, embedding in zip(patch_batch, embedding_batch):
                    db.add_patch(
                        filename=patch["filename"],
                        rect=patch["rect"],
                        embedding=embedding,
                    )
        return db

    def _write(self, data: dict):
        if not self._writeable:
            warnings.warn("PatchDB._write called on non-writeable db")
            return

        if self._fp is None:
            self._fp = gzip.open(self.filename, "at")

        self._fp.write(json.dumps(data, separators=(',', ':')) + "\n")

    def _read_lines(self):
        count = -1
        with io.TextIOWrapper(io.BufferedReader(gzip.open(self.filename))) as fp:
            for line in fp.readlines():

                count += 1
                if self._offset is not None and count < self._offset:
                    continue
                if self._limit is not None and count >= self._limit:
                    break

                yield json.loads(line)

    @classmethod
    def _to_numpy(cls, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            pass
        else:
            x = np.array(x)

        return x.astype(np.float32)


class PatchDBIndex:

    class Patch:
        def __init__(self, index: "PatchDBIndex", filename_id: int, rect: List[int]):
            self.index = index
            self.filename_id = filename_id
            self.rect = rect

        def __repr__(self):
            return f"Patch({self.filename_id}, {self.rect})"

        @property
        def filename(self):
            return self.index._id_to_filename[self.filename_id]

        @property
        def image(self) -> torch.Tensor:
            return self.index.db.image(self.filename)

        @property
        def data(self) -> torch.Tensor:
            image = self.image
            return image[:, self.rect[0]: self.rect[0] + self.rect[2], self.rect[1]: self.rect[1] + self.rect[3]]

        @property
        def patch(self) -> torch.Tensor:
            """
            If `PatchDB.patch_shape` is None, returns the same as `.data`,
            if it is defined, the `.data` patch is resized and the color channels are adjusted.
            """
            image = self.data
            if self.index.db.patch_shape is not None:
                image = set_image_channels(image, self.index.db.patch_shape[0])
                image = VF.resize(image, self.index.db.patch_shape[-2:], interpolation=self.index.db.interpolation)
            return image

    def __init__(self, db: PatchDB):
        self.db = db
        self._filename_to_id: Dict[str, int] = {}
        self._id_to_filename: Dict[int, str] = {}
        self._patches = None
        self._faiss: faiss.IndexFlatIP = None

    @property
    def patches(self) -> List[Patch]:
        self._init()
        return self._patches

    @property
    def size(self):
        self._init()
        return self._faiss.ntotal

    def filenames(self) -> List[str]:
        self._init()
        return list(self._filename_to_id)

    def similar_patches(
            self,
            embedding_or_image: Union[Iterable[float], np.ndarray, torch.Tensor],
            count: int = 1,
            with_distance: bool = False,
    ) -> List[List[Union[Patch, Tuple[Patch, float]]]]:

        embedding_or_image = self.db._to_numpy(embedding_or_image)
        if embedding_or_image.ndim == 1:
            embedding = embedding_or_image.reshape(1, -1)
        elif embedding_or_image.ndim == 2:
            embedding = embedding_or_image
        elif embedding_or_image.ndim in (3, 4):
            if embedding_or_image.ndim == 3:
                image_batch = embedding_or_image.reshape(1, *embedding_or_image.shape)
            else:
                image_batch = embedding_or_image

            if self.db.encoder is None:
                raise ValueError(
                    "PatchDBIndex.similar_patches() was passed an image vector but no PatchDB.encoder is defined"
                )
            else:
                with torch.no_grad():
                    embedding = self.db.encoder.encode_image(torch.Tensor(image_batch))
                    embedding = self.db._to_numpy(embedding / embedding.norm())
        else:
            raise ValueError(
                f"PatchDBIndex.similar_patches() expects 1-4 dim vector, got {embedding_or_image.ndim}"
            )
        self._init()

        distances_batch, indices_batch = self._faiss.search(embedding, count)

        if with_distance:
            return [
                [
                    (self.patches[idx], dist)
                    for idx, dist in zip(indices, distances)
                ]
                for indices, distances in zip(indices_batch, distances_batch)
            ]
        else:
            return [
                [
                    self.patches[idx]
                    for idx in indices
                ]
                for indices in indices_batch
            ]

    def _init(self):
        if self._patches is not None:
            return

        embedding_batch = []
        self._patches = []
        for patch in self.db.iter_patches():
            filename = patch["filename"]
            if filename not in self._filename_to_id:
                self._filename_to_id[filename] = filename_id = len(self._filename_to_id)
                self._id_to_filename[filename_id] = filename
            else:
                filename_id = self._filename_to_id[filename]

            self._patches.append(self.Patch(
                index=self,
                filename_id=filename_id,
                rect=patch["rect"],
            ))

            embedding_batch.append(patch["embedding"])

            if len(embedding_batch) >= 10_000:
                self._add_embeddings_faiss(embedding_batch)
                embedding_batch.clear()

        if embedding_batch:
            self._add_embeddings_faiss(embedding_batch)

    def _add_embeddings_faiss(self, embedding_batch):
        if self._faiss is None:
            self._faiss = faiss.IndexFlatIP(len(embedding_batch[0]))

        embedding_batch = np.concatenate([e.reshape(1, -1) for e in embedding_batch])
        self._faiss.add(embedding_batch)
