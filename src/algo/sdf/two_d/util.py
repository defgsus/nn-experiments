import math
from typing import Union, Tuple, List, Optional, Iterable

import numpy as np


Vec2d = Union[int, float, Tuple[float, float], List[float], np.ndarray]

DTYPE = np.float64
EPS = 1e-10


def vec2_to_numpy(x: Vec2d) -> np.ndarray:
    if isinstance(x, (int, float)):
        x = (x, x)

    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=DTYPE)

    if x.shape != (2,):
        raise ValueError(f"Expected 2 numbers, got {x.shape}")

    return x[None, None, :]


def mix(a: np.ndarray, b: np.ndarray, x: Union[float, np.ndarray]):
    if isinstance(x, np.ndarray):
        while x.ndim < a.ndim:
            x = x[..., None]

    return a * (1. - x) + x * b


def smooth(x):
    return np.pow(x, 2) * (3. - 2. * x)

def smooth2(x):
    return np.pow(x, 3) * (x * (x * 6. - 15.) + 10.)

def step(x, mi, ma):
    return np.clip((x - mi) / (ma - mi), 0, 1)

def smooth_step(x, mi, ma):
    return smooth(step(x, mi, ma))

def smooth2_step(x, mi, ma):
    return smooth2(step(x, mi, ma))


def magnitude(space: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.pow(space, 2.).sum(axis=-1)
    )


def normalize(x: np.ndarray) -> np.ndarray:
    m = magnitude(x)[..., None]
    return x / (m + EPS)


def rotate_space(space: np.ndarray, degree: float):
    theta = degree / 180. * math.pi
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    m = np.array(
        [
            [cos_t,-sin_t],
            [sin_t, cos_t],
        ],
        dtype=space.dtype,
    )

    return np.dot(space, m)


def gain(x: np.ndarray, exp: float):
    """Exponential gain for [0, 1] values"""
    x_mask = x >= .5
    a = x.copy()
    a[x_mask] = 1. - a[x_mask]
    a = .5 * np.pow(2 * a, exp)
    a[x_mask] = 1. - a[x_mask]
    return a


def calc_light(
        normal: np.ndarray,
        light_normal: Vec2d = (-1, 0),
):
    if normal.ndim != 3 or normal.shape[-1] not in (2, 3):
        raise ValueError(f"Expected shape [H, W, 2|3], got {normal.shape}")
    if normal.shape[-1] == 3:
        normal = normal[..., :2]
    light_normal = normalize(vec2_to_numpy(light_normal))
    light = (light_normal * normal).sum(-1)
    return light


def perlin_noise_2d(
        shape: Tuple[int, int],
        res: Tuple[int, int],
        wrap: Tuple[bool, bool] = (False, False),
        rng: Optional[np.random.Generator] = None,
        constant_edge: Optional[float] = None,
) -> np.ndarray:
    from src.algo.perlin_np import numpy_perlin_noise_2d

    noise_shape = tuple(
        ((shape[i] + res[i] - 1) // res[i]) * res[i]
        for i in range(2)
    )
    noise = numpy_perlin_noise_2d(
        shape=noise_shape,
        res=res,
        warp=wrap,
        rng=rng,
        constant_edge=constant_edge,
    )
    if noise_shape != shape:
        # TODO: remove torch dependency
        import torch
        import torchvision.transforms.functional as VF
        noise = (
            VF.resize(torch.from_numpy(noise).unsqueeze(0), shape, VF.InterpolationMode.BICUBIC)
            [0].numpy()
        )

    return noise
