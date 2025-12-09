"""
Ambient Occlusion dataset
"""
import random
from pathlib import Path

import torch
from multipledispatch.dispatcher import source
from tqdm import tqdm
import torchvision.transforms.functional as VF
import PIL.Image

from src.config import PROJECT_PATH
from scripts.ao_dataset.render_base import create_font_image, get_distance_field
from scripts.ao_dataset.render_gl import render_shader
from src.util import to_torch_device
from src.models.ca import TotalCALayer

DATASET_PATH = PROJECT_PATH / "datasets" / "ao"
AA = 2


def render_source_images(
        path: Path = DATASET_PATH,
        size: int = 1024,
        aa: int = AA,
        num_images: tuple[int, int] = (50, 10),
        seed: int = 23,
        sizes: tuple[tuple[float, ...]] = (
            (1/23, 1/17, 1/15),
            (1/20, 1/17, 1/15, 1/6),
            (1/20, 1/15, 1/6),
            (1/21, 1/19, 1/3),
            (1/16, 1/14, 1/12, 1/10, 1/8, 1/7, 1/6, 1/5, 1/4),
            (1/6, 1/3),
            (1/22, 1/19, 1/5, 1/2),
        )
):
    rng = random.Random(seed)

    for j, dataset in enumerate(("train", "validation")):
        for i in range(num_images[j]):

            fn_source = path / dataset / "source" / f"{i:04}.png"
            fn_source_big = path / dataset / "source-big" / f"{i:04}.png"

            if not fn_source.exists():
                print(f"{dataset} {i+1}/{num_images[j]} source image {size}(x{aa})")
                si = rng.choice(sizes)
                source_image = create_font_image(
                    size=size * aa,
                    aa=1,
                    seed=rng.randrange(2**32),
                    desired_mean=rng.uniform(0.5, 1) * max(.1, sum(si) / len(si)),
                    sizes=si,
                )
                fn_source.parent.mkdir(parents=True, exist_ok=True)
                VF.to_pil_image(source_image).resize((size, size), PIL.Image.Resampling.BICUBIC).save(fn_source)

                fn_source_big.parent.mkdir(parents=True, exist_ok=True)
                VF.to_pil_image(source_image).save(fn_source_big)


torch.no_grad()
def render_target_images(
        path: Path = DATASET_PATH,
        device: str = "auto",
        aa: int = AA,
):
    device = to_torch_device(device)
    ca = TotalCALayer(
        birth=[0, 1, 1, 0, 0, 0, 0, 0, 0],
        survive=[0, 1, 1, 1, 1, 1, 0, 0, 0],
        iterations=10,
        threshold=.01,
    ).to(device)
    for dataset in ("train", "validation"):
        files = sorted((path / dataset / "source-big").glob("*.png"))
        for fn_idx, fn_source in enumerate(files):

            fn_distance = path / dataset / "distance" / fn_source.with_suffix(".pt").name
            fn_target = path / dataset / "target" / fn_source.name

            source_image = VF.to_tensor(PIL.Image.open(fn_source))[0]

            if fn_distance.exists():
                distances = torch.load(fn_distance)
            else:
                print(f"{dataset} {fn_idx+1}/{len(files)} distance image {source_image.shape}")

                distances = get_distance_field(source_image.to(device)).cpu()
                fn_distance.parent.mkdir(parents=True, exist_ok=True)
                torch.save(distances, fn_distance)

            if 1 or not fn_target.exists():
                target_size = [source_image.shape[0] // aa, source_image.shape[1] // aa]
                print(f"{dataset} {fn_idx+1}/{len(files)} target image {target_size}")
                d = distances / distances.shape[0]
                d = VF.resize(d.unsqueeze(0), target_size, VF.InterpolationMode.BICUBIC)[0]
                db = VF.gaussian_blur(d.unsqueeze(0), [target_size[0]//64+1]*2, [20., 20.])[0]
                s = VF.resize(source_image.unsqueeze(0), target_size, VF.InterpolationMode.BICUBIC)[0]
                sb = VF.gaussian_blur(s.unsqueeze(0), [target_size[0]//8+1]*2, [20., 20.])[0]
                #sb = ca(s.unsqueeze(0).to(device))[0].cpu()
                image = render_shader(torch.stack([d, db, s, sb]))
                fn_target.parent.mkdir(parents=True, exist_ok=True)
                image.save(fn_target)

            #break


if __name__ == "__main__":
    render_source_images()
    render_target_images()
