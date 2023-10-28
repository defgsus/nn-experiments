from typing import Iterable, Generator, Callable

import torch
from tqdm import tqdm


def iter_batches(source: Iterable, batch_size: int) -> Generator:
    assert batch_size >= 1, f"Batch size must be >= 1, got {batch_size}"

    def _iter_lists():
        item_list = []
        for item in source:
            item_list.append(item)
            if len(item_list) == batch_size:
                yield item_list
                item_list.clear()

        if item_list:
            yield item_list

    def _join(items: list):
        item = items[0]
        if isinstance(item, torch.Tensor):
            return torch.concat([i.unsqueeze(0) for i in items])
        else:
            return items

    for item_list in _iter_lists():
        item = item_list[0]
        is_tuple = isinstance(item, (list, tuple))

        if not is_tuple:
            yield _join(item_list)
        else:
            yield tuple(
                _join([item[item_idx] for item in item_list])
                for item_idx in range(len(item))
            )


def batch_call(
        func: Callable[[torch.Tensor], torch.Tensor],
        data: torch.Tensor,
        batch_size: int = 64,
        verbose: bool = False,
):
    """
    Call a function with a tensor, divide tensor into batches.

    :param func: any callable that expects a single Tensor argument
    :param data: Tensor with ndim >= 2, batched along the first dimension
    :param batch_size: int, size of one batch
    :param verbose: bool, show progress
    :return: Tensor, the concatenated batches
    """
    with tqdm(total=data.shape[0], disable=not verbose) as progress:
        results = []
        for batch_idx in range(0, data.shape[0], batch_size):
            batch_data = data[batch_idx: batch_idx + batch_size]
            results.append(func(batch_data))
            progress.update(batch_data.shape[0])

        return torch.concat(results)
