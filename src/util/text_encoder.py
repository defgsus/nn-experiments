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

from src.util import to_torch_device, iter_batches
from src.util.embedding import normalize_embedding


class TextEncoder:

    def __init__(
            self,
            model_name: str,
            device: str = "auto",
            trust_remote_code: bool = False,
            batch_size: int = 256,
    ):
        """
        Initialize the TextEncoder wrapper.

        :param model_name: str, can be one of
            - `clip/<name>`: The OpenAI CLIP text encoder, e.g. `clip/ViT-B/32`
            - `hug/<name>`: A huggingface transformer, e.g. `thenlper/gte-small`
            - `bow/<length>`: A bag-of-words with maximum length
            - `boc/<length>`: A bag-of-characters with maximum length
            - `bytefreq`: A histogram of character/byte counts in range [0, 0xFF]
        :param device: str or device, the devise, if model is a torch model
        :param trust_remote_code: bool, allow huggingface transformers to execute code locally
        :param batch_size: int, maximum batch size for encoding texts
            text encoding will be batched if the texts are larger than this size
        """
        self.model_name = model_name
        self.device = to_torch_device(device)
        self.trust_remote_code = trust_remote_code
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def encode(self, texts: Iterable[str], normalize: bool = True) -> torch.Tensor:
        embeddings = []
        for batch in iter_batches(texts, batch_size=self.batch_size):
            embeddings.append(self._encode(batch, normalize))
        return torch.concat(embeddings)

    def _encode(self, texts: Iterable[str], normalize: bool = True) -> torch.Tensor:
        if self.model_name == "bytefreq":
            return self._encode_bytes(texts, normalize=normalize)

        elif self.model_name.startswith("clip/"):
            return self._encode_clip(texts, normalize=normalize)

        elif self.model_name.startswith("bow/") or self.model_name.startswith("boc/"):
            return self._encode_bow(texts, normalize=normalize)

        elif self.model_name.startswith("hug/"):
            pass

        else:
            raise ValueError(f"Unrecognized model '{self.model_name}'")

        max_length = None
        if self.model_name == "hug/thenlper/gte-small":
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
        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModel

        model_name = self.model_name[4:]  # remove the 'hug/' part

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=self.trust_remote_code)
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=self.trust_remote_code).to(self.device)

    def _encode_bow(self, texts: Iterable[str], normalize: bool) -> torch.Tensor:
        size = int(self.model_name.split("/")[-1])

        tensors = []
        for text in texts:
            bow = dict()

            if self.model_name.startswith("boc"):
                for ch in text.lower():
                    bow[ch] = bow.get(ch, 0) + 1
            else:
                for word in text.lower().split():
                    bow[word] = bow.get(word, 0) + 1

            values = sorted(bow.values(), reverse=True)
            if len(values) < size:
                values.extend([0] * (size - len(values)))
            elif len(values) > size:
                values = values[:size]

            tensors.append(torch.Tensor(values).unsqueeze(0))

        tensors = torch.concat(tensors)
        if normalize:
            tensors = normalize_embedding(tensors)

        return tensors

    def _encode_bytes(self, texts: Iterable[str], normalize: bool) -> torch.Tensor:
        tensors = []
        for text in texts:

            values = [0] * 256
            for ch in text.encode():
                values[ch] += 1

            tensors.append(torch.Tensor(values).unsqueeze(0))

            # Unfortunately, the numpy version is much slower..
            # byte_array = np.frombuffer(text.encode(), dtype=np.uint8)
            # hist = np.histogram(byte_array, 256, (0, 256))[0]

        tensors = torch.concat(tensors)
        if normalize:
            tensors = normalize_embedding(tensors)

        return tensors
