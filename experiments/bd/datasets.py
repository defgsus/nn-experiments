import gzip
import json
import dataclasses
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import torchvision.datasets
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from src.datasets import *
from src.algo.boulderdash import *
from src.transforms import *


@dataclasses.dataclass
class BDEntry:
    state: torch.Tensor
    next_state: torch.Tensor
    action: Optional[torch.Tensor] = None
    reward: Optional[torch.Tensor] = None

    def __len__(self):
        return self.state.shape[0]

    def to(self, device):
        self.state = self.state.to(device)
        self.next_state = self.next_state.to(device)
        if self.action is not None:
            self.action = self.action.to(device)
        if self.reward is not None:
            self.reward = self.reward.to(device)
        return self


def bd_collate(batch):
    if isinstance(batch, list) and batch and isinstance(batch[0], BDEntry):
        return BDEntry(
            state=default_collate([i.state for i in batch]),
            next_state=default_collate([i.next_state for i in batch]),
            action=None if batch[0].action is None else default_collate([i.action for i in batch]),
            reward=None if batch[0].reward is None else default_collate([i.reward for i in batch]),
        )

    return default_collate(batch)


def boulderdash_dataset_32x32(
        validation: bool,
        diverse: bool = False,
        signed: bool = True,
):
    diverse = "-diverse" if diverse else ""
    path = Path(__file__).resolve().parent.parent.parent / "datasets"
    if validation:
        filename_part = f"boulderdash-32x32-5000{diverse}-validation"
    else:
        filename_part = f"boulderdash-32x32-60000{diverse}"

    ds = WrapDataset(TensorDataset(
        torch.load(path / f"{filename_part}-map1.pt"),
        torch.load(path / f"{filename_part}-map2.pt"),
    )).transform(dtype=torch.float, transform_all=True)
    if signed:
        ds = ds.transform([lambda x: x * 2 - 1.], transform_all=True)
    return ds


def boulderdash_example_dataset_16x16(
        validation: bool,
        signed: bool = True,
):
    """
    `python experiments/bd_rl create_examples examples01-1M -c 1_000_000`
    """
    class _Dataset(BaseIterableDataset):
        def __len__(self):
            return 1_000_000

        def __iter__(self):
            with gzip.open(Path(__file__).resolve().parent.parent / "bd_rl/data/examples01-1M.ndjson.gz") as fp:
                while line := fp.readline():
                    data = json.loads(line)

                    state = BoulderDash.from_array(data["state"]).to_tensor()

                    for action in data["actions"]:
                        if action["action"] == BoulderDash.ACTIONS.Nop:
                            next_state = BoulderDash.from_array(action["next_state"]).to_tensor()

                            yield state, next_state

    ds = _Dataset()
    if validation:
        ds = ds.limit(5000)
    else:
        ds = ds.skip(5000).shuffle(1000)

    if signed:
        ds = ds.transform([lambda x: x * 2 - 1.], transform_all=True)

    return ds


def boulderdash_action_predict_dataset_16x16(
        validation: bool,
        signed: bool = True,
        with_diamond_path: bool = False,
):
    to_tensor_kwargs = {}
    if signed:
        to_tensor_kwargs["zero"] = -1

    class _Dataset(BaseIterableDataset):
        def __len__(self):
            return 1_000_000 * BoulderDash.ACTIONS.count()

        def __iter__(self):
            with gzip.open(Path(__file__).resolve().parent.parent / "bd_rl/data/examples02-1M.ndjson.gz") as fp:
                while line := fp.readline():
                    data = json.loads(line)

                    state = BoulderDash.from_array(data["state"]).to_tensor(**to_tensor_kwargs)

                    for action in data["actions"]:
                        next_state = BoulderDash.from_array(action["next_state"]).to_tensor(**to_tensor_kwargs)
                        action_vec = [-1 if signed else 0] * BoulderDash.ACTIONS.count()
                        action_vec[action["action"]] = 1
                        reward = action["reward"]

                        if with_diamond_path:
                            if data.get("closest_diamond") is not None:
                                if action.get("closest_diamond") is not None:
                                    if len(action["closest_diamond"]) < len(data["closest_diamond"]):
                                        reward += .5

                        yield BDEntry(
                            state=state,
                            next_state=next_state,
                            action=torch.Tensor(action_vec),
                            reward=torch.Tensor([reward]),
                        )

    ds = _Dataset()
    if validation:
        ds = ds.limit(5000)
    else:
        ds = ds.skip(5000).shuffle(1000)

    return ds
