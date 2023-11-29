import argparse
import datetime
import gzip
import itertools
import json
import base64
import os
from pathlib import Path
from typing import Iterable

from tqdm import tqdm
import torch
import torch.nn.functional as F

from src.util import to_torch_device
from src.util.embedding import normalize_embedding


class TextEncoder:

    def __init__(
            self,
            model_name: str,
            device: str = "auto",
            trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        self.device = to_torch_device(device)
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

    def encode(self, texts: Iterable[str], normalize: bool = True) -> torch.Tensor:
        if self.model_name.startswith("clip/"):
            return self._encode_clip(texts, normalize=normalize)

        max_length = None
        if self.model_name == "thenlper/gte-small":
            max_length = 512

        self._get_model()
        tokens = self._tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        tokens = {
            key: value.to(self.device)
            for key, value in tokens.items()
        }
        model_output = self._model(**tokens)
        embedding = self._mean_pooling(model_output, tokens['attention_mask'])
        if normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        import torch

        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _encode_clip(self, texts: Iterable[str], normalize: bool) -> torch.Tensor:
        from src.models.clip import ClipSingleton
        embedding = ClipSingleton.encode_text(texts, truncate=True, model=self.model_name.split("/", 1)[-1])
        if normalize:
            embedding = normalize_embedding(embedding)
        return embedding

    def _get_model(self):
        if self.model_name.startswith("clip/"):
            return

        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModel

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        self._model = AutoModel.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code).to(self.device)
