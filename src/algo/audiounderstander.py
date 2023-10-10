import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torchaudio.transforms as AT
import torchaudio.functional as AF

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from src.util.image import iter_image_patches


class AudioUnderstander:
    """
    Based on:

    Attention is All You Need?
    Good Embeddings with Statistics are Enough: Large Scale Audio Understanding without Transformers/ Convolutions/ BERTs/ Mixers/ Attention/ RNNs or ....
    Prateek Verma
    https://browse.arxiv.org/pdf/2110.03183.pdf
    """
    def __init__(
            self,
            sample_rate: int = 44100,
            slice_size: int = 44100,
            spectral_shape: Tuple[int, int] = (128, 128),
            spectral_patch_shapes: Iterable[Tuple[int, int]] = ((8, 8), ),
            encoder_ratios: Iterable[int] = (8, ),
            num_clusters: Iterable[int] = (256, ),
    ):
        self.sample_rate = sample_rate
        self.slice_size = slice_size
        self.spectral_shape = tuple(spectral_shape)
        self.spectral_patch_shapes = [tuple(s) for s in spectral_patch_shapes]
        self.encoder_ratios = list(encoder_ratios)
        self.num_clusters = list(num_clusters)
        self._spectrogrammer = AT.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024 * 2,
            win_length=sample_rate // 30,
            hop_length=sample_rate // spectral_shape[-1],
            n_mels=spectral_shape[-2],
            power=1.,
        )
        self.encoders = [
            nn.Linear(math.prod(spectral_patch_shape), math.prod(spectral_patch_shape) // encoder_ratio)
            for spectral_patch_shape, encoder_ratio in zip(self.spectral_patch_shapes, self.encoder_ratios)
        ]
        self.clusterers = [
            MiniBatchKMeans(
                n_clusters=num_clusters,
                batch_size=1024,
                random_state=23,
                n_init=1,
            )
            for num_clusters in self.num_clusters
        ]

        for check_attribute in ("encoder_ratios", "clusterers"):
            if len(self.spectral_patch_shapes) != len(getattr(self, check_attribute)):
                raise ValueError(
                    f"`{check_attribute}` must be same length as `spectral_patch_shapes`"
                    f", expected {len(self.spectral_patch_shapes)}, got {len(getattr(self, check_attribute))}"
                )

    @torch.inference_mode()
    def encode_audio(self, audio: torch.Tensor, normalize_spec: bool = True) -> torch.Tensor:
        if audio.ndim == 1:
            pass
        elif audio.ndim == 2:
            if audio.shape[0] > 1:
                audio = audio.mean(0)
            else:
                audio = audio.squeeze(0)
        else:
            raise ValueError(f"Need audio.ndim == 1 or 2, got {audio.ndim}")

        if audio.shape[-1] < self.slice_size:
            audio = torch.concat([audio, torch.zeros(self.slice_size - audio.shape[-1]).to(audio.dtype)])

        histograms = []
        while audio.shape[-1] >= self.slice_size:
            spec = self._get_spec(audio[:self.slice_size], normalize=normalize_spec)

            histograms.append(self._get_histogram(spec))

            audio = audio[self.slice_size:]

        return torch.concat(histograms)

    @torch.inference_mode()
    def encode_spectrum(self, spectrum: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if spectrum.ndim == 2:
            pass
        else:
            raise ValueError(f"Need spectrum.ndim == 2, got {spectrum.ndim}")

        if spectrum.shape[-2] != self.spectral_shape[0]:
            raise ValueError(f"spectrum.shape must fit `spectral_shape`, expected {self.spectral_shape}, got {spectrum.shape}")

        if spectrum.shape[-1] < self.spectral_shape[-1]:
            spectrum = torch.concat([spectrum, torch.zeros(spectrum.shape[-2], self.spectral_shape[-1] - spectrum.shape[-1]).to(spectrum.dtype)])

        if normalize:
            spec_max = spectrum.max()
            if spec_max:
                spectrum = spectrum / spec_max

        histograms = []
        while spectrum.shape[-1] >= self.spectral_shape[-1]:
            spec = spectrum[:, :self.spectral_shape[-1]]
            histograms.append(self._get_histogram(spec))

            spectrum = spectrum[:, self.spectral_shape[-1]:]

        return torch.concat(histograms)

    def drop_encoder(self, index: int):
        self.spectral_patch_shapes.pop(index)
        self.encoders.pop(index)
        self.clusterers.pop(index)

    def _get_histogram(self, spec: torch.Tensor) -> torch.Tensor:
        one_full_hist = []
        for spectral_patch_shape, encoder, clusterer in zip(self.spectral_patch_shapes, self.encoders, self.clusterers):
            patches = torch.concat(list(iter_image_patches(spec.unsqueeze(0), spectral_patch_shape)))

            embeddings = encoder(patches.flatten(1))

            labels = clusterer.predict(embeddings.flatten(1))

            hist, _ = np.histogram(labels, bins=clusterer.n_clusters, range=(0, clusterer.n_clusters - 1))

            one_full_hist.append(torch.Tensor(hist) / patches.shape[0])

        return torch.concat(one_full_hist).unsqueeze(0)

    def _get_spec(self, audio: torch.Tensor, normalize: bool):
        spec = self._spectrogrammer(audio)[:, :self.spectral_shape[-1]]
        if normalize:
            spec_max = spec.max()
            if spec_max:
                spec = spec / spec_max
        return spec.clamp(0, 1)

    def save(self, file) -> None:
        data = {
            "sample_rate": self.sample_rate,
            "slice_size": self.slice_size,
            "spectral_shape": self.spectral_shape,
            "spectral_patch_shapes": self.spectral_patch_shapes,
            "encoder_ratios": self.encoder_ratios,
            "num_clusters": self.num_clusters,
        }
        for i in range(len(self.clusterers)):
            data.update({
                f"encoder.{i}.weight": self.encoders[i].weight[:],
                f"encoder.{i}.bias": self.encoders[i].bias[:],
                f"clusterer.{i}": self.clusterers[i]
            })
        torch.save(data, file)

    @classmethod
    def load(cls, fp):
        data = torch.load(fp)
        c = cls(
            sample_rate=data["sample_rate"],
            slice_size=data["slice_size"],
            spectral_shape=data["spectral_shape"],
            spectral_patch_shapes=data["spectral_patch_shapes"],
            encoder_ratios=data["encoder_ratios"],
            num_clusters=data["num_clusters"],
        )
        with torch.no_grad():
            for i in range(len(c.clusterers)):
                c.encoders[i].weight[:] = data[f"encoder.{i}.weight"]
                c.encoders[i].bias[:] = data[f"encoder.{i}.bias"]
                c.clusterers[i] = data[f"clusterer.{i}"]
        return c
