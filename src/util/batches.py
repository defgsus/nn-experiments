from typing import Iterable, Generator

import torch


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
