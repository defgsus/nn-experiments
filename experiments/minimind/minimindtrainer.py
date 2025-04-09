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
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src import console
from src.util.image import signed_to_image, get_images_from_iterable
from src.util import FontSquares
from src.train import Trainer


class MiniMindTrainer(Trainer):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            max_seq_length: int,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self._font_squares = FontSquares(shape=(3, 8, 8))
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length

    def train_step(self, batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        X, Y, loss_mask = batch

        output = self.model(X)
        loss = F.cross_entropy(
            output.logits.reshape(-1, output.logits.size(-1)),
            Y.reshape(-1),
            reduction="none",
        ).view(Y.size())

        loss = (loss * loss_mask).sum() / loss_mask.sum()
        loss += output.aux_loss
        # loss = loss / output.accumulation_steps

        with torch.no_grad():
            return {
                "loss": loss,
                "error%": (Y != output.logits.argmax(dim=-1)).float().mean() * 100,
            }

    def write_step(self):
        return

        def _example(name: str, batch_iterable):
            for batch in batch_iterable:
                text_batch, masked_batch = self._split_batch(batch)
                break
            text_batch = text_batch[:64]
            masked_batch = masked_batch[:64]

            output_logits = self.model(masked_batch)
            output_batch = output_logits.argmax(dim=-1)

            num_correct = (text_batch == output_batch).sum(-1)
            index = num_correct.argsort()

            grid = []
            #for text, output, mask in zip(text_batch, output_batch, masked_batch):
            for idx in reversed(index):
                text = text_batch[idx]
                mask = masked_batch[idx]
                output = output_batch[idx]

                r_text = self._font_squares(text, dim=-1)
                r_mask = self._font_squares(mask, dim=-1)
                r_output = self._font_squares(output, dim=-1)

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

            self.log_image(f"image_{name}", make_grid(grid, nrow=1, padding=2))

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
    def generate_text(
            self,
            start: Union[str, torch.Tensor],
            length: int = 200,
            num_c: int = 1,
            keep_length: bool = True,
    ) -> torch.Tensor:
        if isinstance(start, str):
            text_batch = self._encode_texts([start])
        else:
            text_batch = start.to(self.device)

        generated_text_batch = text_batch.clone()
        while generated_text_batch.shape[-1] < length:
            text_batch_with_mask = torch.concat([
                generated_text_batch[:, -text_batch.shape[-1]:] if keep_length else generated_text_batch,
                torch.zeros(text_batch.shape[0], num_c, dtype=text_batch.dtype).to(self.device)
            ], dim=-1)

            output_logits = self.model(text_batch_with_mask)
            output_text_batch = output_logits[:, -num_c:].argmax(dim=-1)
            # print("LOGITS", output_logits.shape, output_text_batch.shape)

            generated_text_batch = torch.concat([
                generated_text_batch,
                output_text_batch
            ], dim=-1)

        return generated_text_batch
