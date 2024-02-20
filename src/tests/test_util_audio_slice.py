import torch

from src.tests.base import *
from src.util.audio import *


class TestAudioSlice(TestBase):

    def get_audio_stream(self, length: int, chunk_size: int, channels: int):
        audio = torch.randn(length, channels)
        while audio.shape[0]:
            yield audio[:chunk_size]
            audio = audio[chunk_size:]

    def test_100_no_stride(self):
        for chunk_size in (30, 1000, 3000, 3001, 50000):
            for channels in (1, 2):
                self.assertEqual(
                    [(3000, channels), (3000, channels), (3000, channels), (1000, channels)],
                    [s.shape for s in iter_audio_slices(
                        stream=self.get_audio_stream(length=10_000, chunk_size=chunk_size, channels=channels),
                        slice_size=3_000,
                    )]
                )
                self.assertEqual(
                    [(3000, channels), (3000, channels), (3000, channels)],
                    [s.shape for s in iter_audio_slices(
                        stream=self.get_audio_stream(length=10_000, chunk_size=chunk_size, channels=channels),
                        slice_size=3_000,
                        max_slices=3,
                    )]
                )
                self.assertEqual(
                    [],
                    [s.shape for s in iter_audio_slices(
                        stream=self.get_audio_stream(length=10_000, chunk_size=chunk_size, channels=channels),
                        slice_size=3_000,
                        max_slices=0,
                    )]
                )

    def test_200_stride(self):
        for chunk_size in (30, 1000, 3000, 3001, 50000):
            for channels in (1, 2):
                self.assertEqual(
                    [(3000, channels)] * 8 + [(2000, channels), (1000, channels)],
                    [s.shape for s in iter_audio_slices(
                        stream=self.get_audio_stream(length=10_000, chunk_size=chunk_size, channels=channels),
                        slice_size=3_000,
                        stride=1_000,
                    )]
                )
