import random
from typing import Union, Tuple

import torch


def random_combine_image_crops(
        images: torch.Tensor,
        ratio: float = .5,
        crop_ratio: Union[float, Tuple[float, float]] = .5,
        alpha: Union[float, Tuple[float, float]] = 1.,
):
    if ratio <= 0.:
        return images

    crop_ratio = (crop_ratio, crop_ratio) if isinstance(crop_ratio, (float, int)) else tuple(crop_ratio)
    alpha = (alpha, alpha) if isinstance(alpha, (float, int)) else tuple(alpha)
    ret_images = []

    for image_idx, image in enumerate(images):
        if random.random() > ratio:
            ret_images.append(image)
        else:
            while True:
                other_idx = random.randrange(images.shape[0])
                if other_idx != image_idx:
                    break
            other_image = images[other_idx]

            crop_size = [
                random.uniform(*crop_ratio)
                for i in range(2)
            ]
            crop_size = [
                max(1, min(int(c * image.shape[i + 1]), image.shape[i + 1] - 1))
                for i, c in enumerate(crop_size)
            ]
            source_pos = [random.randrange(0, s - crop_size[i]) for i, s in enumerate(other_image.shape[-2:])]
            target_pos = [random.randrange(0, s - crop_size[i]) for i, s in enumerate(other_image.shape[-2:])]

            ss_y = slice(source_pos[0], source_pos[0] + crop_size[0])
            ss_x = slice(source_pos[1], source_pos[1] + crop_size[1])
            ts_y = slice(target_pos[0], target_pos[0] + crop_size[0])
            ts_x = slice(target_pos[1], target_pos[1] + crop_size[1])
            a = random.uniform(*alpha)

            image = image.clone()
            if a >= 1.:
                image[:, ts_y, ts_x] = other_image[:, ss_y, ss_x]
            else:
                image[:, ts_y, ts_x] += a * (other_image[:, ss_y, ss_x] - image[:, ts_y, ts_x])

            ret_images.append(image)

    return torch.concat([i.unsqueeze(0) for i in ret_images], dim=0)
