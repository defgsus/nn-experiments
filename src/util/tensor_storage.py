import os
from pathlib import Path
from typing import Optional, Union, List

import torch


class TensorStorage:
    def __init__(
            self,
            filename_part: Union[str, Path],
            max_items: Optional[int] = None,
            max_bytes: Optional[int] = None,
            num_zero_padding_digits: int = 6,
    ):
        self.filename_part = Path(filename_part)
        self.filename_index = 0
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.num_zero_padding_digits = num_zero_padding_digits
        self.shape: Optional[torch.Size] = None
        self.dtype: Optional[torch.dtype] = None
        self.buffer: List[torch.Tensor] = []
        self.buffer_bytes = 0


    @property
    def filename(self):
        return self.filename_part.parent / f"{self.filename_part.name}-{self.filename_index:0{self.num_zero_padding_digits}}.pt"

    def add(self, tensor: torch.Tensor):
        if self.dtype is None:
            self.dtype = tensor.dtype
        else:
            if self.dtype != tensor.dtype:
                raise ValueError(f"Expected tensor with dtype {self.dtype}, got {self.dtype}")
        if self.shape is None:
            self.shape = tensor.shape
        else:
            if self.shape != tensor.shape:
                raise ValueError(f"Expected tensor of shape {self.shape}, got {tensor.shape}")

        num_bytes = tensor.nelement() * tensor.element_size()

        if self.max_bytes is not None and (self.buffer_bytes + num_bytes) >= self.max_bytes:
            self.store_buffer()

        self.buffer.append(tensor)
        self.buffer_bytes += num_bytes

        if self.max_items is not None and len(self.buffer) >= self.max_items:
            self.store_buffer()

    def store_buffer(self):
        if not self.buffer:
            return

        os.makedirs(self.filename_part.parent, exist_ok=True)
        torch.save(torch.stack(self.buffer), self.filename)
        self.buffer.clear()
        self.buffer_bytes = 0
        self.filename_index += 1
