import math
import random

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def create_font_image(
        size: int = 512,
        desired_mean: float = 0.05,
        seed: int = 23,
        aa: int = 1,
        fonts: tuple[str] = (
                "/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf",
                "/usr/share/fonts/opentype/linux-libertine/LinLibertine_RB.otf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ),
        sizes: tuple[float, ...] = (1/15, 1/10, 1/5, 1/3),
):
    rng = random.Random(seed)

    size_aa = size * aa
    img = torch.zeros((size_aa, size_aa), dtype=torch.float32)
    pil_fonts = [
        PIL.ImageFont.truetype(f, size=int(size_aa * s))
        for s in sizes
        for f in fonts
    ]
    def _iter_fonts():
        while True:
            yield rng.choice(pil_fonts)
            idx = rng.randrange(rng.randrange(len(pil_fonts)) + 1)
            yield pil_fonts[idx]

    for font in _iter_fonts():
        text = chr(rng.randrange(33, 127))
        box = [int(b) for b in font.getbbox(text=text)]
        fimg = PIL.Image.new("L", (box[2] - box[0], box[3] - box[1]))
        draw = PIL.ImageDraw.ImageDraw(fimg)
        draw.text(xy=(-box[0], -box[1]), text=text, font=font, fill=255)
        fimg = fimg.rotate(angle=rng.uniform(0, 360), expand=True)
        mask = PIL.Image.new("L", (size_aa, size_aa))
        xy = (rng.randrange(size_aa - fimg.width), rng.randrange(size_aa - fimg.height))
        mask.paste(im=fimg, box=xy)
        mask = VF.to_tensor(mask)[0]
        img = torch.max(img, mask)
        if img.mean() >= desired_mean:
            break

    if aa > 1:
        img = VF.resize(img.unsqueeze(0), [size, size], VF.InterpolationMode.BICUBIC)[0]
    return img.clamp(0, 1)



@torch.no_grad()
def get_distance_field(bitmap: torch.Tensor, dtype=torch.float):
    """
    Calculate min distance field for bitmap

    - https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.102.7988&rep=rep1&type=pdf
    - https://www.shadertoy.com/view/4lKGDt

    :param bitmap: Tensor,
    :param dtype:
    :return:
    """
    assert bitmap.ndim == 2, f"Got {bitmap.shape}"
    e1 = 1
    e2 = math.sqrt(2)
    max_d = math.sqrt(math.prod(bitmap.shape)) + 1

    distances = torch.ones_like(bitmap).to(dtype) * max_d
    positions = torch.ones((bitmap.shape[0], bitmap.shape[1], 2), dtype=torch.int64, device=bitmap.device) * -1
    xy = torch.tensor(np.mgrid[:bitmap.shape[0], :bitmap.shape[1]], device=bitmap.device).permute(1, 2, 0).to(torch.int64)

    # assign immediate boundary points
    y_mask = bitmap[1:] != bitmap[:-1]
    x_mask = bitmap[:, 1:] != bitmap[:, :-1]
    mask = (
        # pad (left, right, top, bottom)
          F.pad(y_mask, (0, 0, 0, 1))
        | F.pad(y_mask, (0, 0, 1, 0))
        | F.pad(x_mask, (0, 1, 0, 0))
        | F.pad(x_mask, (1, 0, 0, 0))
    )
    positions[mask] = xy[mask]
    distances[mask] = 0

    # iterate
    last_changes = -1
    num_same_changes = 0
    with tqdm(range(max(bitmap.shape))) as progress:
        for it in progress:
            scale = 1 if it > 20 else random.uniform(1, min(bitmap.shape))
            num_changes = 0
            for sy, sx, isy, isx, pad, e in (
                    # TL
                    (slice(None, -1), slice(None, -1), slice(1, None), slice(1, None), (1, 0, 1, 0), e2),
                    # T
                    (slice(None, -1), slice(None, None), slice(1, None), slice(None, None), (0, 0, 1, 0), e1),
                    # TR
                    (slice(None, -1), slice(1, None), slice(1, None), slice(None, -1), (0, 1, 1, 0), e2),
                    # L
                    (slice(None, None), slice(None, -1), slice(None, None), slice(1, None), (1, 0, 0, 0), e1),
                    # R
                    (slice(None, None), slice(1, None), slice(None, None), slice(None, -1), (0, 1, 0, 0), e1),
                    # BL
                    (slice(1, None), slice(None, -1), slice(None, -1), slice(1, None), (1, 0, 0, 1), e2),
                    # B
                    (slice(1, None), slice(None, None), slice(None, -1), slice(None, None), (0, 0, 0, 1), e1),
                    # BR
                    (slice(1, None), slice(1, None), slice(None, -1), slice(None, -1), (0, 1, 0, 1), e2),
            ):
                mask = F.pad(distances[sy, sx] + (e * scale) - distances[isy, isx], pad) < 0
                num_changes += mask.sum().item()
                positions[mask] = F.pad(positions[sy, sx], (0, 0) + pad, value=-1)[mask]
                distances[mask] = (xy[mask] - positions[mask]).pow(2).sum(-1).sqrt()

            if num_changes == last_changes:
                num_same_changes += 1
            last_changes = num_changes

            progress.set_postfix({"changes": num_changes})
            if num_same_changes >= 5:
                print("BREAK @", it)
                break

    distances[bitmap > 0] *= -1
    return distances
