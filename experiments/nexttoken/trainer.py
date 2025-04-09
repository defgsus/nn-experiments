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


class TextTrainer(Trainer):
    """
    Expects:
        dataset: tokens, expected-tokens

    """
    def __init__(
            self,
            seq_length: int,
            vocab_size: int,
            tokenizer: Optional[PreTrainedTokenizerBase],
            **kwargs,
    ):
        super().__init__(**kwargs)
        self._seq_length = seq_length
        self._tokenizer = tokenizer
        self._vocab_size = vocab_size
        self._font_squares = FontSquares(shape=(3, 8, 8))

    def train_step(self, batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        input_tokens, expected_tokens = batch

        expected_logits = self._token_ids_to_logits(expected_tokens)

        expected_tokens = expected_tokens.flatten(0)

        output_logits = self.model(input_tokens)
        output_tokens = output_logits.argmax(dim=-1)
        # print("XXX", expected_tokens.shape, output_tokens.shape, output_logits.shape, )

        loss = F.cross_entropy(
            output_logits.flatten(1),
            expected_tokens,
        ) + F.l1_loss(output_logits, expected_logits.flatten(1))

        #print("EXPECTED:\n", expected_tokens, "\n", self.decode(expected_tokens))
        #print("GOT:\n", output_tokens, "\n", self.decode(output_tokens))

        with torch.no_grad():
            error_batch = (expected_tokens != output_tokens).float()
            return {
                "loss": loss,
                "error%": error_batch.mean() * 100,
                "logit_mean": output_logits.mean(),
            }

    def _token_ids_to_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        :param tokens: tensor of [B, L]
        :return: tensor of [B, L, <num-tokens>]
        """
        B, L = tokens.shape
        logits = torch.zeros((B, L, self._vocab_size)).to(tokens)
        logits.scatter_(-1, tokens.unsqueeze(-1).to(torch.int64), 1)
        return logits.float()

    def decode(self, token_ids: torch.LongTensor) -> Union[str, List[str]]:
        single_dim = False
        if token_ids.ndim == 1:
            single_dim = True
            token_ids = token_ids.unsqueeze(0)

        texts = []
        for ids in token_ids:
            texts.append(
                "".join(self._tokenizer.convert_ids_to_tokens(ids))
                .replace("⬇", " ").replace("⬅", "\n")
            )
        if single_dim:
            return texts[0]
        return texts

    def write_step(self):
        pass
