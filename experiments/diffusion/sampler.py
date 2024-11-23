from typing import List, Iterable, Tuple, Optional, Callable, Union, Dict

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF


class DiffusionSamplerBase:

    def create_noise_amounts(
            self,
            batch_size: int,
            minimum: float = 0.001,
            maximum: float = 1.,
            seed: Union[None, int, torch.Generator] = None,
    ) -> torch.Tensor:
        gen = self._to_generator(seed)

        amounts = torch.rand((batch_size, 1), generator=gen)
        if minimum == maximum:
            return amounts * maximum
        return amounts * ((maximum - minimum) + minimum)

    def add_noise(
            self,
            images: torch.Tensor,
            noise_amounts: Optional[torch.Tensor] = None,
            seed: Union[None, int, torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise, return (noisy-images, noise-amounts)
        """
        assert images.ndim == 4, f"Got {images.shape}"
        batch_size = images.shape[0]

        gen = self._to_generator(seed)

        if noise_amounts is None:
            noise_amounts = self.create_noise_amounts(batch_size, seed=gen)

        return (
            self._add_noise(images, noise_amounts.to(images), gen),
            noise_amounts,
        )

    def remove_noise(
            self,
            images: torch.Tensor,
            noise: torch.Tensor,
            noise_amounts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert images.ndim == 4, f"Got {images.shape}"

        if noise_amounts is not None:
            noise = noise * noise_amounts[:, None, None]

        return (images - noise).clamp(-1, 1)

    def _add_noise(
            self,
            images: torch.Tensor,
            noise_amounts: torch.Tensor,
            generator: Optional[torch.Generator],
    ):
        raise NotImplementedError

    def _to_generator(self, seed: Union[None, int, torch.Generator]) -> Optional[torch.Generator]:
        if seed is None:
            return None #torch.Generator()
        elif isinstance(seed, torch.Generator):
            return seed
        else:
            return torch.Generator().manual_seed(seed)


class DiffusionSamplerNoise(DiffusionSamplerBase):

    def _add_noise(
            self,
            images: torch.Tensor,
            noise_amounts: torch.Tensor,
            generator: Optional[torch.Generator],
    ):
        noise = torch.randn(images.shape, generator=generator).to(images)
        noise = noise * noise_amounts[:, None, None]
        return (images + noise).clamp(-1, 1)


class DiffusionSamplerBlur(DiffusionSamplerBase):

    def __init__(
            self,
            kernel_size: int,
            sigma: float,
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def _add_noise(
            self,
            images: torch.Tensor,
            noise_amounts: torch.Tensor,
            generator: Optional[torch.Generator],
    ):
        sigmas = noise_amounts.flatten().pow(1.7) * self.sigma
        return torch.cat([
            VF.gaussian_blur(
                image,
                kernel_size=[self.kernel_size, self.kernel_size],
                sigma=[float(sigma), float(sigma)],
            ).unsqueeze(0)
            if sigma > 0 else image
            for image, sigma in zip(images, sigmas)
        ])
