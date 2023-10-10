from typing import Iterable, Union, Tuple, Optional, Generator

import torch


def iter_audio_slices(
        stream: Iterable[Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        slice_size: int,
        channels: Optional[int] = None,
        stride: Optional[int] = None,
        max_slices: Optional[int] = None,
        with_position: bool = False,
) -> Generator[Union[torch.Tensor, Tuple[torch.Tensor, int]], None, None]:

    if stride is None:
        stride = slice_size

    buffer = None
    buffer_start = 0
    num_slices = 0

    for chunk in stream:
        if isinstance(chunk, (tuple, list)):
            chunk = chunk[0]

        if channels is not None:
            chunk = chunk[..., :channels]

        if buffer is None:
            buffer = chunk
        else:
            buffer = torch.concat([buffer, chunk])

        while buffer.shape[0] >= slice_size:
            if max_slices is not None and num_slices >= max_slices:
                break

            if with_position:
                yield buffer[:slice_size], buffer_start
            else:
                yield buffer[:slice_size]
            num_slices += 1

            buffer = buffer[stride:]
            buffer_start += stride

        if max_slices is not None and num_slices >= max_slices:
            break

    if buffer is None:
        return

    while buffer.shape[0]:
        if max_slices is not None and num_slices >= max_slices:
            break

        if with_position:
            yield buffer[:slice_size], buffer_start
        else:
            yield buffer[:slice_size]
        num_slices += 1

        buffer = buffer[stride:]
        buffer_start += stride
