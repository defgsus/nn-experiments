import torch
from torch.utils.data import DataLoader
import torchvision.transforms as VT

from tqdm import tqdm

from src.tests.base import *
from src.util.image import *
from src.util import iter_parameter_permutations, param_make_tuple


class TestImagePatches(TestBase):

    def test_map_image_patches(self):
        for params in tqdm(list(iter_parameter_permutations({
            "image_shape": (
                (3, 64, 64),
                (1, 32, 64),
                (1, 64, 31),
                (2, 11, 10),
            ),
            "patch_size": (
                32,
                (31, 33),
                50,
            ),
            "overlap": (
                0,
                1,
                (10, 9),
                (14, 17),
                (31, 31),
            ),
            "cut_away": (
                0,
                1,
                (2, 3),
            ),
            "window": (False, True),
        }))):
            msg = ", ".join(
                f"{key}={repr(value)}"
                for key, value in params.items()
            )
            # print(msg)

            image_shape = params["image_shape"]
            patch_size = params["patch_size"]
            overlap = params["overlap"]
            cut_away = params["cut_away"]
            window = params["window"]

            if (
                any(o >= s for o, s in zip(param_make_tuple(overlap, 2), param_make_tuple(patch_size, 2)))
                or any(o + c >= s for o, c, s in zip(
                    param_make_tuple(overlap, 2), param_make_tuple(cut_away, 2), param_make_tuple(patch_size, 2)
                ))
            ):
                continue

            image = torch.zeros(*image_shape)

            try:
                output = map_image_patches(
                    image=image,
                    function=lambda x: 1. - x,  # invert the patches
                    patch_size=patch_size,
                    overlap=overlap,
                    cut_away=cut_away,
                    batch_size=32,
                    auto_pad=True,
                    window=window,
                )

                self.assertEqual(
                    image.shape,
                    output.shape,
                )

                # test that all of the image is inverted
                self.assertEqual(
                    1.,
                    output.mean().item(),
                )

            except Exception as e:
                e.args = (*e.args, msg)
                raise
