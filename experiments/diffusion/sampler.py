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
            noise_amounts = self.create_noise_amounts(batch_size, seed=gen).to(images)

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

        return images - noise

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


class DiffusionSamplerNoiseMix(DiffusionSamplerBase):

    def _add_noise(
            self,
            images: torch.Tensor,
            noise_amounts: torch.Tensor,
            generator: Optional[torch.Generator],
    ):
        noise = torch.rand(images.shape, generator=generator).to(images) * 2. - 1.
        a = noise_amounts[:, None, None]
        return noise * a + (1. - a) * images


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


class DiffusionSamplerNoiseBlur(DiffusionSamplerBase):

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
        noise = torch.randn(images.shape, generator=generator).to(images)
        noise = noise * noise_amounts[:, None, None]
        images = (images + noise).clamp(-1, 1)

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


class DiffusionSamplerDeform(DiffusionSamplerBase):

    def __init__(
            self,
            alpha: float = 50.,
            sigma: float = 3.,
            fill: float = -1.,
    ):
        self.alpha = alpha
        self.sigma = sigma
        self.fill = fill

    @staticmethod
    def get_displacement(sigma: List[float], size: List[int], batch_size, generator):
        dx = torch.rand([1, 1] + size, generator=generator) * 2 - 1
        if sigma[0] > 0.0:
            kx = int(8 * sigma[0] + 1)
            # if kernel size is even we have to make it odd
            if kx % 2 == 0:
                kx += 1
            dx = VF.gaussian_blur(dx, [kx, kx], sigma)
        dx = dx / (size[0] / batch_size)

        dy = torch.rand([1, 1] + size, generator=generator) * 2 - 1
        if sigma[1] > 0.0:
            ky = int(8 * sigma[1] + 1)
            # if kernel size is even we have to make it odd
            if ky % 2 == 0:
                ky += 1
            dy = VF.gaussian_blur(dy, [ky, ky], sigma)
        dy = dy / size[1]
        return torch.concat([dx, dy], 1).permute([0, 2, 3, 1])[0]  # 1 x H x W x 2

    def _add_noise(
            self,
            images: torch.Tensor,
            noise_amounts: torch.Tensor,
            generator: Optional[torch.Generator],
    ):
        B, C, H, W = images.shape
        disp = self.get_displacement([self.sigma, self.sigma], [B * H, W], B, generator).to(images)
        # disp is [B * H, W, 2]
        disp = disp.view(B, H, W, 2) * noise_amounts[:, None, None] * self.alpha

        return torch.cat([
            VF.elastic_transform(image.unsqueeze(0), d.unsqueeze(0), fill=self.fill)
            for image, d in zip(images, disp)
        ])
