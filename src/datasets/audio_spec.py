import warnings
import random
from pathlib import Path
from typing import Union, Optional, Iterable, Tuple

import torch
from torchaudio.io import StreamReader
import torchaudio.transforms as AT
from torch.utils.data import IterableDataset

from src.util.files import iter_filenames
from src.util.audio import iter_audio_slices, is_audio_file
from .base_iterable import BaseIterableDataset


# TODO: make this use an audio-slice dataset as source
class AudioSpecIterableDataset(BaseIterableDataset):

    def __init__(
            self,
            path: Union[str, Path] = "~/Music/",
            recursive: bool = False,
            sample_rate: int = 44100,
            slice_size: int = 44100,
            stride: Optional[int] = None,
            interleave_files: Optional[int] = None,
            shuffle_files: bool = False,
            shuffle_slices: Optional[int] = None,
            mono: bool = False,
            seek_offset: float = 0.,
            with_filename: bool = False,
            with_position: bool = False,

            spec_slice_size: int = 44100,
            spec_shape: Tuple[int, int] = (64, 64),
            spec_stride: int = 1,
    ):
        from scripts import datasets

        self.spec_shape = spec_shape
        self.spec_slice_size = spec_slice_size
        self.spec_stride = spec_stride
        self.sample_rate = sample_rate
        self.with_filename = with_filename
        self.with_position = with_position
        self.slice_ds = datasets.audio_slice_dataset(
            path=path,
            recursive=recursive,
            interleave_files=interleave_files,
            slice_size=slice_size,
            stride=stride,
            mono=mono,
            shuffle_files=shuffle_files,
            shuffle_slices=shuffle_slices,
            seek_offset=seek_offset,
            with_position=True,
            with_filename=True,
        )
        self.speccer = AT.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024 * 2,
            win_length=self.spec_slice_size // spec_shape[-1],
            hop_length=self.spec_slice_size // spec_shape[-1],
            n_mels=spec_shape[-2],
            power=1.,
        )

    def __iter__(self):
        for audio, fn, pos in self.slice_ds:
            spec = self.speccer(audio)

            for offset in range(0, spec.shape[-1], self.spec_stride):
                # print(offset, self.spec_shape[-1], spec.shape[-1])
                if offset + self.spec_shape[-1] <= spec.shape[-1]:
                    audio_offset = int(offset / spec.shape[-1] * audio.shape[-1])
                    # print("X", audio_offset, audio_slice_width, audio.shape[-1])
                    if audio_offset + self.spec_slice_size <= audio.shape[-1]:
                        spec_slice = spec[..., offset:offset + self.spec_shape[-1]]
                        audio_slice = audio[..., audio_offset:audio_offset + self.spec_slice_size]

                        tup = [audio_slice, spec_slice]
                        if self.with_filename:
                            tup.append(fn)
                        if self.with_position:
                            tup.append(pos + audio_offset)

                        yield tup
