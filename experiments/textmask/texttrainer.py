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
from src.util.image import signed_to_image, get_images_from_iterable
from src.util import FontSquares
from src.train import Trainer


class TextMaskTrainer(Trainer):
    """
    Expects:
        dataset: str items

    """
    def __init__(
            self,
            *args,
            mask_is_arg: Optional[int] = None,
            mask_size: int = 10,
            mask_type: Union[str, Iterable[str]] = "single",  # "block", "single"
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._mask_is_arg = mask_is_arg
        self._mask_size = mask_size
        self._mask_type = [mask_type] if isinstance(mask_type, str) else set(mask_type)
        self._font_squares = FontSquares(shape=(3, 8, 8))

    def _split_batch(self, batch):
        if self._mask_is_arg is not None:
            texts = self._encode_texts(batch[0])
            masks = self._encode_texts(batch[self._mask_is_arg])
        else:
            texts = batch
            if isinstance(batch, (tuple, list)) and not isinstance(batch[0], str):
                texts = batch[0]
            texts = self._encode_texts(texts)
            masks = self._apply_mask(texts)

        return texts, masks

    def train_step(self, batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        text_batch, masked_batch = self._split_batch(batch)

        output_logits = self.model(masked_batch)

        target_logits = self._encoded_to_logits(text_batch)

        loss = F.cross_entropy(output_logits.view(-1, 256), target_logits.view(-1, 256).float())
        #loss = F.l1_loss(output_logits, target_logits.float())
        with torch.no_grad():
            mask_mask = (masked_batch == 0)
            return {
                "loss": loss,
                "error%": (text_batch != output_logits.argmax(dim=-1)).float().mean() * 100,
                "mask_error%": (
                    mask_mask & (text_batch != output_logits.argmax(dim=-1))
                ).float().mean() * 100 / mask_mask.float().mean(),
            }

    def _apply_mask(self, texts: torch.Tensor) -> torch.Tensor:
        """
        :param text: tensor of shape [B, L]
        :return: masked tensor
        """
        B, L = texts.shape
        mask = torch.ones(B, L, dtype=texts.dtype).to(texts.device)
        if "block" in self._mask_type:
            size = max(1, min(L, self._mask_size))
            indices = torch.randint(0, L - size, (B, 1)).to(texts.device)
            coords = torch.arange(0, L).unsqueeze(0).repeat(B, 1).to(texts.device)
            mask &= (coords < indices) | (coords >= indices + size)
        if "single" in self._mask_type:
            for i in range(self._mask_size):
                mask.scatter_(-1, torch.randint(0, L - 1, (B, 1)).to(texts.device), 0)
        #else:
        #    raise NotImplementedError(f"No mask type '{self._mask_type}'")

        return texts * mask

    def _encode_texts(self, texts: List[str]):
        text_batch = []
        texts = [list(t.encode()) for t in texts]
        max_len = max(len(t) for t in texts)
        for text in texts:
            if len(text) < max_len:
                text += [32] * (max_len - len(text))

            text_batch.append(
                torch.tensor(text, dtype=torch.int)
                .unsqueeze(0)
            )

        return torch.cat(text_batch).to(self.device)

    def _encoded_to_logits(self, texts: torch.Tensor) -> torch.Tensor:
        """
        :param texts: tensor of [B, L]
        :return: tensor of [B, L, 256]
        """
        B, L = texts.shape
        logits = torch.zeros((B, L, 256)).to(texts)
        logits.scatter_(-1, texts.unsqueeze(-1).to(torch.int64), 1)
        return logits

    def write_step(self):
        def _example(name: str, batch_iterable):
            for batch in batch_iterable:
                text_batch, masked_batch = self._split_batch(batch)
                break
            text_batch = text_batch[:64]
            masked_batch = masked_batch[:64]

            output_logits = self.model(masked_batch)
            output_batch = output_logits.argmax(dim=-1)

            grid = []
            for text, output, mask in zip(text_batch, output_batch, masked_batch):

                r_text = self._font_squares(text, dim=-1)
                r_mask = self._font_squares(mask, dim=-1)
                r_output= self._font_squares(output, dim=-1)

                for i, (t, o, m) in enumerate(zip(text, output, mask)):
                    if t == o:
                        r_output[i, 1] = r_output[i, 1].clamp_min(.4)
                    else:
                        r_output[i, 0] = r_output[i, 0].clamp_min(.618)

                grid.append(make_grid([
                    make_grid(r_text, padding=0, nrow=r_text.shape[0]),
                    make_grid(r_mask, padding=0, nrow=r_mask.shape[0]),
                    make_grid(r_output, padding=0, nrow=r_output.shape[0]),
                ], nrow=1))

            self.log_image(f"image_{name}", make_grid(grid, nrow=2, padding=2))

            texts = self.generate_text(text_batch[:5][:text_batch.shape[-1] // 3])
            grid = []
            for text in texts:
                grid.append(make_grid(
                    [
                        self._font_squares(ch)
                        for ch in text
                    ],
                    nrow=60, padding=0,
                ))
            self.log_image(f"image_generated_{name}", make_grid(grid, nrow=1, padding=4))

        _example("train", self.iter_training_batches())
        _example("validation", self.iter_validation_batches())

    @torch.no_grad()
    def generate_text(self, start: Union[str, torch.Tensor], length: int = 200, num_c: int = 1) -> torch.Tensor:
        if isinstance(start, str):
            text_batch = self._encode_texts([start])
        else:
            text_batch = start.to(self.device)
        while text_batch.shape[-1] < length:
            text_batch_with_mask = torch.concat([
                text_batch,
                torch.zeros(text_batch.shape[0], num_c, dtype=text_batch.dtype).to(self.device)
            ], dim=-1)

            output_logits = self.model(text_batch_with_mask)
            output_text_batch = output_logits[:, -num_c:].argmax(dim=-1)
            # print("LOGITS", output_logits.shape, output_text_batch.shape)
            text_batch = torch.concat([
                text_batch,
                output_text_batch
            ], dim=-1)

        return text_batch
