import math
import os
import random
import time
from pathlib import Path
from typing import Iterable, Tuple, Union

import moderngl
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import torchvision.transforms.functional as VF

from scripts.extrusion_dataset.mesh import TriangleMesh
from scripts.extrusion_dataset import curve2d
from scripts.extrusion_dataset.render import render_meshes, get_light_map


def render_image(
        size: int = 2048,
        radius_pixels: int = 5,
        seed: int = 23,
        num_curves: int = 100,
        num_copies: int = 10,
):
    rng = random.Random(seed)
    radius_fac = 2. * radius_pixels / size

    def create_curve():
        points = []
        p = (0, 0)
        step = tuple(
            max(0.01, math.pow(rng.uniform(0, 1), 2) * .2)
            for i in range(2)
        )
        for i in range(rng.randint(2, 9)):
            points.append(p)
            p = (
                p[0] + rng.uniform(-1, 1) * step[0],
                p[1] + rng.uniform(-1, 1) * step[1],
            )

        if rng.uniform(0, 1) < .5:
            radius = (.2 + .8 * math.pow(rng.uniform(0, 1), 2)) * radius_fac
        else:
            freq = rng.uniform(.1, 1) * (len(points) - 1)

            def _radius(t: float):
                r = .5 + .5 * math.sin(t * freq * math.pi * 2)
                r = .2 + .8 * math.pow(r, 2)
                return r * radius_fac

            radius = _radius

        curve_class = rng.choices([
            curve2d.CurveLinear2d,
            curve2d.CurveHermite2d,
        ], [1, .7])[0]
        return curve_class(points), radius

    def yield_meshes():
        for i in tqdm(range(num_curves), desc="mashing the curves"):
            curve, radius = create_curve()
            mesh = curve.to_mesh(radius, point_distance=2 / size)
            for i in range(num_copies):
                sc = rng.uniform(.7, 1.3)
                yield (
                    mesh.centered_at(random.uniform(-1, 1), random.uniform(-1, 1))
                    .rotated_z(random.uniform(0, 360))
                    .scaled(sc, sc, sc)
                )

    return render_meshes(yield_meshes(), size)


def render_dataset(
        output_path: Path = Path(__file__).resolve().parent.parent / "datasets" / "extrusion",
):
    def store_images(subpath, index, image_source, image_target):
        for image, subsubpath in ((image_source, "source"), (image_target, "target")):
            path = output_path / subpath / subsubpath
            os.makedirs(path, exist_ok=True)
            image.save(path / f"{index:03}.png")

    index = 0
    for seed in (23, 42, 66, 101):
        for num_curves in (50, 100):
            image_low, image_high = render_image(
                seed=seed,
                num_curves=num_curves,
            )
            store_images("train", index, image_low, image_high)
            index += 1

    image_low, image_high = render_image(
        seed=27362873,
        num_curves=75,
    )
    store_images("validation", index, image_low, image_high)


def test():
    image_low, image_high = render_image()

    image_low.save(Path(__file__).resolve().parent / "test-low.png")
    image_high.save(Path(__file__).resolve().parent / "test-high.png")


def test_speed():
    from tqdm import tqdm
    mesh = TriangleMesh()
    for i in tqdm(range(1000000)):
        mesh.add_triangle((-1, -1, 0), (1, -.9, 0), (0, 1, 0))


def render_light_gif(
        normal_filename=Path(__file__).resolve().parent / "test-high.png",
        frames: int = 100,
):
    normals = PIL.Image.open(normal_filename)
    normals = VF.to_tensor(normals).numpy() * 2 - 1
    normals = normals[:, 256:512, 256:512]

    images = []
    for i in tqdm(range(frames), desc="lighting"):
        t = i / frames * math.pi * 2
        x = math.sin(t) * 2
        y = math.cos(t) * 2
        z = 3
        light_map = get_light_map(normals, (x, y, z))
        images.append(VF.to_pil_image(torch.Tensor(light_map)))

    images[0].save(
        Path(__file__).resolve().parent / "test-light.gif",
        save_all=True, append_images=images[1:],
        optimize=False, duration=1 / 15, loop=0,
    )


if __name__ == "__main__":
    #test_speed()
    #test()
    #render_light_gif()
    render_dataset()
