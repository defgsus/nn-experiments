import warnings
import random
from pathlib import Path
from typing import Union, Optional, Iterable

import torch
from torchaudio.io import StreamReader
from torch.utils.data import IterableDataset

from src.util.files import iter_filenames
from src.util.audio import iter_audio_slices, is_audio_file


class AudioSliceIterableDataset(IterableDataset):

    def __init__(
            self,
            path: Union[str, Path, Iterable[Union[str, Path]]],
            sample_rate: int,
            slice_size: int,
            stride: Optional[int] = None,
            recursive: bool = False,
            interleave_files: Optional[int] = None,
            shuffle_files: bool = False,
            max_slices_per_file: Optional[int] = None,
            mono: bool = False,
            sample_order: str = "CxS",  # "CxS" or "SxC"
            seek_offset: float = 0.,
            with_filename: bool = False,
            with_position: bool = False,
            verbose: bool = False,
    ):
        assert sample_order in ("CxS", "SxC"), f"Got {sample_order}"

        self.path = path
        self.recursive = bool(recursive)
        self.sample_rate = int(sample_rate)
        self.slice_size = int(slice_size)
        self.stride = self.slice_size if stride is None else int(stride)
        self.interleave_files = 1 if interleave_files is None else int(interleave_files)
        self.shuffle_files = bool(shuffle_files)
        self.max_slices_per_file = max_slices_per_file
        self.mono = bool(mono)
        self.sample_order = sample_order
        self.seek_offset = seek_offset
        self.with_filename = bool(with_filename)
        self.with_position = bool(with_position)
        self.verbose = bool(verbose)
        self._filenames = None

    def __iter__(self):
        if self._filenames is None:
            self._filenames = sorted(
                f for f in iter_filenames(self.path, recursive=self.recursive)
                if is_audio_file(f)
            )

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            filenames = self._filenames
        else:
            filenames = self._filenames[worker_info.id::worker_info.num_workers]

        if self.shuffle_files:
            filenames = filenames.copy()
            random.shuffle(filenames)

        streams = []
        while True:
            while filenames and len(streams) < self.interleave_files:
                filename = filenames[0]
                filenames = filenames[1:]

                try:
                    if self.verbose:
                        print(f"{self.__class__.__name__}: OPEN {filename}")

                    reader = StreamReader(str(filename))
                    reader.add_basic_audio_stream(4096, sample_rate=self.sample_rate)
                    reader.seek(self.seek_offset)
                except RuntimeError:
                    continue

                streams.append({
                    "filename": filename,
                    "iterable": iter_audio_slices(
                        stream=reader.stream(),
                        slice_size=self.slice_size,
                        channels=None,
                        stride=self.stride,
                        max_slices=self.max_slices_per_file,
                        with_position=self.with_position,
                    ),
                })

            if not streams:
                break

            next_streams = []
            for stream in streams:
                try:
                    chunk = next(stream["iterable"])
                    next_streams.append(stream)
                except StopIteration:
                    if self.verbose:
                        print(f"{self.__class__.__name__}: FINISHED {stream['filename']}")
                    continue
                except RuntimeError:
                    warnings.warn(f"Error reading from stream {stream['filename']}")
                    continue

                if self.with_position:
                    chunk, position = chunk
                else:
                    position = None

                if chunk.shape[0] == self.slice_size:

                    if self.mono:
                        if chunk.ndim == 2:
                            if self.sample_order == "CxS":
                                audio_slice = chunk.mean(1).unsqueeze(0)
                            else:
                                audio_slice = chunk.mean(1).unsqueeze(0).permute(1, 0)
                        else:
                            raise RuntimeError(
                                f"Can't handle chunk.ndim == {chunk.ndim}, shape == {chunk.shape}"
                            )
                    else:
                        if self.sample_order == "CxS":
                            audio_slice = chunk.permute(1, 0)
                        else:
                            audio_slice = chunk

                    if self.with_filename or self.with_position:
                        ret_tuple = [audio_slice]
                        if self.with_filename:
                            ret_tuple.append(stream["filename"])
                        if self.with_position:
                            ret_tuple.append(position)
                        yield tuple(ret_tuple)

                    else:
                        yield audio_slice

            streams = next_streams
