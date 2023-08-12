"""
Inspired by:
https://github.com/hirokatsukataoka16/FractalDB-Pretrained-ResNet-PyTorch
"""
import random
from typing import Optional, Tuple, Generator

import numpy as np


class IFS:
    max_coordinate = 1e10

    def __init__(
            self,
            seed: Optional[int] = None,
            num_parameters: int = 2,
            parameters: Optional[np.ndarray] = None,
            probabilities: Optional[np.ndarray] = None,
    ):
        self.rng = np.random.Generator(np.random.MT19937(
            seed if seed is not None else random.randint(0, int(1e10))
        ))
        self.rng.bytes(100)
        self.parameters = self.rng.uniform(-1., 1., (num_parameters, 6))
        self.probabilities = self.rng.uniform(0., 1., (num_parameters, ))
        if parameters is not None:
            self.parameters = parameters
        if probabilities is not None:
            self.probabilities = probabilities

    def iter_coordinates(self, num_iterations: int) -> Generator[Tuple[float, float], None, None]:
        x, y = 0., 0.
        for iteration in range(num_iterations):
            param_index = None
            while param_index is None:
                idx = self.rng.integers(0, self.parameters.shape[0])
                if self.rng.uniform(0., 1.) < self.probabilities[idx]:
                    param_index = idx

            a, b, c, d, e, f = self.parameters[param_index]

            x, y = (
                x * a + y * b + e,
                x * c + y * d + f
            )

            if np.abs(x) > self.max_coordinate or np.abs(y) > self.max_coordinate:
                #print(f"early stop at iteration {iteration}")
                break

            if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
                yield x, y
            else:
                #print(f"early stop at iteration {iteration}")
                break

    def render_coordinates(self, shape: Tuple[int, int], num_iterations: int, padding: int = 2) -> np.ndarray:
        coords = np.array(list(self.iter_coordinates(num_iterations)))
        min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
        min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
        if max_x != min_x:
            coords[:, 0] = (coords[:, 0] - min_x) / (max_x - min_x) * (shape[-1] - padding * 2) + padding
        if max_y != min_y:
            coords[:, 1] = (coords[:, 1] - min_y) / (max_y - min_y) * (shape[-2] - padding * 2) + padding
        return coords.astype(np.uint16)

    def render_image(
            self,
            shape: Tuple[int, int],
            num_iterations: int,
            padding: int = 2,
            alpha: float = 0.1,
            patch_size: int = 1,
    ) -> np.ndarray:

        extra_padding = 0
        if patch_size > 1:
            extra_padding = patch_size
            shape = (shape[-2] + extra_padding * 2, shape[-1] + extra_padding * 2)

        coords = self.render_coordinates(shape, num_iterations, padding + extra_padding)
        image = np.zeros((1, *shape))

        if patch_size <= 1:
            for x, y in coords:
                image[0, y, x] += alpha

        else:
            half_patch_size = patch_size // 2
            patch = np.hamming(patch_size).repeat(patch_size).reshape(patch_size, patch_size)
            patch *= patch.T * alpha

            for x, y in coords:
                x -= half_patch_size
                y -= half_patch_size
                image[0, y:y + patch_size, x:x + patch_size] += patch


            image = image[:, extra_padding:-extra_padding, extra_padding:-extra_padding]

        return image.clip(0, 1)
