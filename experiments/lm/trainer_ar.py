import math
import random
from typing import List, Iterable, Tuple, Optional, Callable, Union, Dict

import torch
import torch.nn
import torch.fft
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import make_grid

from transformers import AutoTokenizer, AutoModelForCausalLM

from src import console
from src.util.image import signed_to_image, get_images_from_iterable
from src.train import Trainer
from src.models.transform import Sobel


class TrainAutoregressiveLM(Trainer):

    def __init__(
            self,
            *args,
            tokenizer_name: str = "EleutherAI/gpt-neo-125M",
            max_context_length: Optional[int] = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=True)
        self.context_length = self.model.config.max_position_embeddings
        self.max_context_length = max_context_length

    def train_step(self, input_batch) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(input_batch, (tuple, list)):
            input_batch = input_batch[0]

        text_batch = input_batch["text"]

        context_length = self.context_length
        if self.max_context_length:
            context_length = min(context_length, self.max_context_length)

        loss_sum = None
        num_correct = 0

        for text in text_batch:
            token_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            token_ids = token_ids[:, :context_length]

            logits = self.model(token_ids)["logits"]
            # logits = F.softmax(logits, dim=-1)

            expected_logits = torch.zeros_like(logits)
            for i, id in enumerate(token_ids[0, 1:]):
                expected_logits[0, i, id] = 1.

            num_correct += (logits.argmax(dim=-1)[0, :-1] == token_ids[0, 1:]).sum() / token_ids.shape[-1]

            loss = self.loss_function(logits, expected_logits)

            # print(loss, logits, expected_logits)

            if loss_sum is None:
                loss_sum = loss
            else:
                loss_sum = loss_sum + loss

        loss_sum = loss_sum / len(text_batch)
        num_correct /= len(text_batch)

        return {
            "loss": loss_sum,
            "accuracy": num_correct,
        }
