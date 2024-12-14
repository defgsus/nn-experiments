from pathlib import Path
import math
import random
import time
from typing import List, Iterable, Tuple, Optional, Callable, Union, Dict

import PIL.ImageFont, PIL.ImageDraw, PIL.Image
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import torch.utils.data
import torchvision.models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import make_grid

from src import console
from src.util.image import signed_to_image, get_images_from_iterable, set_image_channels
from src.util import FontSquares
from src.train import Trainer


class TextMaskFontTrainer(Trainer):
    """
    Expects:
        dataset: str items

    """
    def __init__(
            self,
            *args,
            mask_size: int = 10,
            font_shape: Tuple[int, int, int] = (1, 8, 8),
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._mask_size = mask_size
        self._font_squares = FontSquares(
            shape=font_shape,
        )

    def train_step(self, batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        texts = batch
        if isinstance(batch, (tuple, list)) and not isinstance(batch[0], str):
            texts = batch[0]

        masked_texts = self._apply_mask(texts)
        target_batch = self._encode_texts(texts)
        masked_batch = self._encode_texts(masked_texts)

        output_batch = self.model(masked_batch)

        loss = F.l1_loss(output_batch, target_batch.float())

        return {
            "loss": loss,
            # "error%": (text_batch != output_logits.argmax(dim=-1)).float().mean() * 100,
        }

    def _encode_texts(self, texts: List[str]):
        text_batch = []
        max_len = max(len(t) for t in texts)
        for text in texts:
            if len(text) < max_len:
                text = text + " " * (max_len - len(text))

            text_batch.append(
                self._font_squares(text, dim=-1)
            )

        return torch.stack(text_batch).to(self.device)

    def _apply_mask(self, texts: List[str]) -> List[str]:
        """
        :param text: tensor of shape [B, L, C, H, W]
        :return: masked tensor
        """
        masked_texts = []
        for text in texts:
            size = max(1, min(len(text) - 1, self._mask_size))
            if len(text) <= size:
                masked_texts.append(text)
            else:
                index = random.randrange(len(text) - size)
                masked_texts.append(
                    text[:index] + "\0" * size + text[index + size:]
                )
        return masked_texts

    def write_step(self):
        for batch in self.iter_validation_batches():
            texts = batch
            if isinstance(batch, (tuple, list)) and not isinstance(batch[0], str):
                texts = batch[0]
            break

        masked_texts = self._apply_mask(texts)
        target_batch = self._encode_texts(texts)
        masked_batch = self._encode_texts(masked_texts)

        output_batch = self.model(masked_batch)

        output_texts = [
            self._font_squares.reverse(o, dim=-1)
            for o in output_batch
        ]
        error = sum(
            sum(1 if c1 != c2 else 0 for c1, c2 in zip(target, output))
            for target, output in zip(texts, output_texts)
        ) / math.prod(target_batch.shape[:2])
        self.log_scalar("validation_error%", error * 100)

        grid = []
        for text, mask_text, out_text, target, output, masked in zip(
                texts, masked_texts, output_texts, target_batch, output_batch, masked_batch,
        ):
            target = set_image_channels(target, 3)
            masked = set_image_channels(masked, 3)
            output = set_image_channels(output, 3)
            for i, (ch, mch, och) in enumerate(zip(text, mask_text, out_text)):
                #if mch == "\0":
                if ch == och:
                    output[i, 1] = output[i, 1].clamp_min(.4)
                else:
                    output[i, 0] = output[i, 0].clamp_min(.4)
            grid.append(make_grid([
                make_grid(target, padding=0, nrow=target.shape[0]),
                make_grid(masked, padding=0, nrow=target.shape[0]),
                make_grid(output, padding=0, nrow=target.shape[0]),
            ], nrow=1))

        self.log_image("image_validation", make_grid(grid, nrow=2, padding=2))

