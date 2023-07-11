import math
import random
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Callable

from tqdm import tqdm
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torch.utils.data import DataLoader

from src.datasets import TotalCADataset


def main(
        shape=(64, 64),
        # num_repetitions: int = 5,
        dtype=torch.uint8,
        output_filename="./datasets/ca-64x64-i10-p05.pt",
        max_megabyte=2_048,
):
    # torch.multiprocessing.set_sharing_strategy('file_system')

    ds = TotalCADataset(
        shape,
        num_iterations=10,
        init_prob=.5,
        wrap=True,
        # num_repetitions=num_repetitions,
        #transforms=[lambda x: x.unsqueeze(0)],
        dtype=dtype,
    )

    # ds = DataLoader(ds, num_workers=5)

    tensor_batch = []
    tensor_size = 0
    last_print_size = 0
    with torch.no_grad():
        for image in tqdm(ds):
            if isinstance(image, (tuple, list)):
                image = image[0]

            if len(image.shape) < 4:
                image = image.unsqueeze(0)

            tensor_batch.append(image)
            tensor_size += math.prod(image.shape) * 1

            if tensor_size - last_print_size > 1024 * 1024 * 10:
                last_print_size = tensor_size

                print(f"size: {tensor_size:,}")

            if tensor_size >= max_megabyte * 1024 * 1024:
                break

        tensor_batch = torch.cat(tensor_batch)
        print(f"saving {tensor_batch.shape} to {output_filename}")
        torch.save(tensor_batch, output_filename)


if __name__ == "__main__":
    main()
