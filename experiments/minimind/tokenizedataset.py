import random
from typing import Sequence, Tuple, Union

import torch

from src.datasets import BaseDataset, BaseIterableDataset


class TokenizeDataset(BaseIterableDataset):

    def __init__(
            self,
            texts: Sequence[str],
            tokenizer,
            max_seq_length: int,
            min_seq_length: int = None,
            batch_size: int = None,
            method: str = "concat",  # "truncate", "fragments", "concat", "concatstride"
            stride: Union[int, Tuple[int, int]] = 1,
            return_types: str = "X,Y,lossmatrix",  # "X,Y,lossmatrix", "X[:-1],X[-1:]"
    ):
        self._texts = texts
        self._tokenizer = tokenizer
        self._min_seq_length = min_seq_length
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._method = method
        self._stride = stride
        self._return_types = return_types

    #    def __len__(self):
    #        return len(self._texts)

    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #from tqdm import tqdm
        if self._method == "truncate":
            iterable = self._iter_truncated()
        elif self._method == "fragments":
            iterable = self._iter_fragments()
        elif self._method == "concat":
            iterable = self._iter_concat()
        elif self._method == "concatstride":
            iterable = self._iter_concat_stride()
        else:
            raise ValueError(f"Unknown method `{self._method}`")

        for token_ids in iterable:
            if token_ids.shape[0]:
                loss_mask = (token_ids != self._tokenizer.pad_token_id)

                if self._return_types == "X,Y,lossmatrix":
                    X = token_ids[:-1]
                    Y = token_ids[1:]
                    loss_mask = loss_mask[1:]
                    yield X, Y, loss_mask
                elif self._return_types == "X[:-1],X[-1:]":
                    yield token_ids[:-1], token_ids[-1:]

    def _iter_truncated(self):
        seq_length = self._max_seq_length
        for i, text in enumerate(self._texts):

            if self._min_seq_length:
                assert self._batch_size, "Must define `batch_size` when defining `min_seq_length`"
                if i % self._batch_size == 0:
                    seq_length = random.randint(self._min_seq_length, self._max_seq_length)

            encoding = self._tokenizer(
                text,
                max_length=seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            yield encoding.input_ids.squeeze()

    def _iter_fragments(self):
        seq_length = self._max_seq_length
        count = 0
        for text in self._texts:
            encoding = self._tokenizer(
                text,
                return_tensors='pt'
            )
            input_ids = encoding.input_ids.squeeze()
            if input_ids.shape[0] == 0:
                continue

            while True:
                if self._min_seq_length is not None:
                    assert self._batch_size, "Must define `batch_size` when defining `min_seq_length`"
                    if count % self._batch_size == 0:
                        seq_length = random.randint(self._min_seq_length, self._max_seq_length)
                count += 1

                if input_ids.shape[0] == seq_length:
                    yield input_ids
                    break

                elif input_ids.shape[0] < seq_length:
                    yield torch.cat([
                        torch.ones((seq_length - input_ids.shape[0], ), dtype=input_ids.dtype) * self._tokenizer.pad_token_id,
                        input_ids
                    ])
                    break

                else:
                    yield input_ids[:seq_length]
                    input_ids = input_ids[seq_length // 2:]

    def _iter_concat(self):
        seq_length = self._max_seq_length
        count = 0
        current_ids = None
        for text in self._texts:
            encoding = self._tokenizer(
                text,
                return_tensors='pt'
            )
            input_ids = encoding.input_ids.squeeze()
            if input_ids.shape[0] == 0:
                continue

            while True:
                if self._min_seq_length is not None:
                    assert self._batch_size, "Must define `batch_size` when defining `min_seq_length`"
                    if count % self._batch_size == 0:
                        seq_length = random.randint(self._min_seq_length, self._max_seq_length)

                if input_ids.shape[0] == seq_length:
                    yield input_ids
                    count += 1
                    break

                elif input_ids.shape[0] < seq_length:
                    if current_ids is None:
                        current_ids = input_ids
                    else:
                        current_ids = torch.cat([
                            current_ids,
                            torch.ones((1, ), dtype=input_ids.dtype) * self._tokenizer.sep_token_id,
                            input_ids
                        ])

                    if current_ids.shape[0] >= seq_length:
                        yield current_ids[:seq_length]
                        count += 1
                        current_ids = current_ids[seq_length:]
                    break

                else:
                    yield input_ids[:seq_length]
                    count += 1
                    input_ids = input_ids[seq_length // 2:]

    def _iter_concat_stride(self):
        seq_length = self._max_seq_length
        count = 0
        current_ids = None
        for text in self._texts:
            encoding = self._tokenizer(
                text,
                return_tensors='pt'
            )
            token_ids = encoding.input_ids.squeeze()
            if token_ids.shape[0] == 0:
                continue

            if current_ids is None:
                current_ids = token_ids
            else:
                current_ids = torch.cat([
                    current_ids,
                    torch.ones((1, ), dtype=token_ids.dtype) * self._tokenizer.sep_token_id,
                    token_ids
                ])

            while current_ids.shape[0] >= seq_length:
                yield current_ids[:seq_length]
                count += 1
                if self._min_seq_length is not None:
                    assert self._batch_size, "Must define `batch_size` when defining `min_seq_length`"
                    if count % self._batch_size == 0:
                        seq_length = random.randint(self._min_seq_length, self._max_seq_length)

                stride = self._get_stride()
                current_ids = current_ids[stride:]

    def _get_stride(self) -> int:
        if isinstance(self._stride, int):
            return self._stride
        else:
            return random.randint(self._stride[0], self._stride[1])
