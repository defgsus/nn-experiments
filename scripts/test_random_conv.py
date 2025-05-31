import io
import time
import math
import random
from pathlib import Path
from typing import List, Union, Optional, Tuple, Generator

import torch
import torch.nn as nn
import torchvision.datasets
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from tqdm import tqdm

from src.models.util import activation_to_module
from src import config as global_config
from src.util.binarydb import BinaryDB


class ConvModel(nn.Module):

    def __init__(
            self,
            channels: List[int],
            kernel_size: List[int],
            stride: List[int],
            dilation: List[int],
            activation: List[Optional[str]],
    ):
        super().__init__()

        self.layers = nn.Sequential()
        for i in range(len(channels)):
            self.layers.append(
                nn.Conv2d(
                    in_channels=3 if i == 0 else channels[i - 1],
                    out_channels=channels[i],
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                    dilation=dilation[i],
                )
            )
            if activation[i]:
                self.layers.append(activation_to_module(activation[i]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvModelTester:

    def __init__(
            self,
            model_count: int = 1000,
            layers_range: Tuple[int, int] = (3, 3),
            channel_choices: Tuple[int, ...] = (32, 48, 64),
            kernel_size_choices: Tuple[int, ...] = (3, 5, 7, 9),
            stride_choices: Tuple[int, ...] = (1, 2, 3),
            dilation_choices: Tuple[int, ...] = (1, 2, 3),
            activation_choices: Tuple[Optional[str], ...] = ("relu", ),
            input_shape: Tuple[int, int, int] = (3, 96, 96),
            ratio_range: Tuple[float, float] = (.1, .6),
            seed: int = 42,
            dataset_max_size_train: int = 1000,
            dataset_max_size_test: int = 1000,
            num_random_trails: int = 5,
            cache_path: Path = global_config.PROJECT_PATH / "cache" / "random_pca",
    ):
        self.model_count = model_count
        self.layers_range = layers_range
        self.channel_choices = channel_choices
        self.kernel_size_choices = kernel_size_choices
        self.stride_choices = stride_choices
        self.dilation_choices = dilation_choices
        self.activation_choices = activation_choices
        self.input_shape = input_shape
        self.ratio_range = ratio_range
        self.rng = random.Random(seed)
        self.dataset_max_size_train = dataset_max_size_train
        self.dataset_max_size_test = dataset_max_size_test
        self.num_random_trails = num_random_trails
        self.cache_path = cache_path
        self._dataset = None
        self._db = BinaryDB(self.cache_path / "db2.sqlite")

    @torch.no_grad()
    def iter_model_params(
        self,
    ) -> Generator[dict, None, None]:
        params_hash_set = set()
        num_duplicates = 0
        num_skipped = 0
        with tqdm(total=self.model_count, desc="creating model params") as progress:
            while progress.n < self.model_count:
                progress.set_postfix({"duplicates": num_duplicates, "skipped": num_skipped})

                num_layers = self.rng.randint(*self.layers_range)
                params = {
                    "channels": [self.rng.choice(self.channel_choices) for _ in range(num_layers)],
                    "kernel_size": [self.rng.choice(self.kernel_size_choices) for _ in range(num_layers)],
                    "stride": [self.rng.choice(self.stride_choices) for _ in range(num_layers)],
                    "dilation": [self.rng.choice(self.dilation_choices) for _ in range(num_layers)],
                    "activation": [self.rng.choice(self.activation_choices) for _ in range(num_layers)],
                }
                params_hash = str(params)
                if params_hash in params_hash_set:
                    num_duplicates += 1
                    continue
                params_hash_set.add(params_hash)

                model = ConvModel(**params)

                try:
                    output_shape = model(torch.ones(*self.input_shape)).shape
                except RuntimeError:
                    num_skipped += 1
                    continue

                ratio = math.prod(output_shape) / math.prod(self.input_shape)
                if not self.ratio_range[0] <= ratio <= self.ratio_range[1]:
                    num_skipped += 1
                    continue

                yield params

                progress.update()

    def get_classification_dataset(self):
        if self._dataset is None:
            print("loading dataset")
            ds = torchvision.datasets.STL10(global_config.SMALL_DATASETS_PATH, split="train")
            train_x, train_y = ds.data.astype(np.float32) / 255., ds.labels            
            train_x, train_y = train_x[:self.dataset_max_size_train], train_y[:self.dataset_max_size_train]
            
            ds = torchvision.datasets.STL10(global_config.SMALL_DATASETS_PATH, split="test")
            test_x, test_y = ds.data.astype(np.float32) / 255., ds.labels
            test_x, test_y = test_x[:self.dataset_max_size_test], test_y[:self.dataset_max_size_test]

            self._dataset = train_x, train_y, test_x, test_y

        return self._dataset

    @torch.no_grad()
    def test_classification(
            self,
            model: nn.Module,
            verbose: bool = False,
    ):
        def _process(images: np.ndarray, batch_size: int = 32):
            result = []
            with tqdm(total=images.shape[0], desc="processing images", disable=not verbose) as progress:
                for i in range((images.shape[0] + batch_size - 1) // batch_size):
                    batch = torch.from_numpy(images[i * batch_size: (i + 1) * batch_size])
                    batch = model(batch)
                    result.append(batch.reshape(batch.shape[0], -1).numpy())
                    progress.update(batch.shape[0])
            return np.concat(result, axis=0)

        train_x, train_y, test_x, test_y = self.get_classification_dataset()
        train_x_org = train_x

        start_time = time.time()
        train_x = _process(train_x)
        test_x = _process(test_x)

        throughput = (train_x.shape[0] + test_x.shape[0]) / (time.time() - start_time)
        compression_ratio = math.prod(train_x.shape) / math.prod(train_x_org.shape)

        classifier = RidgeClassifier()
        if verbose:
            print("fitting classifier")
        start_time = time.time()
        classifier.fit(train_x, train_y)
        train_time = time.time() - start_time
        if verbose:
            print("predicting validation set")
        predicted_train_y = classifier.predict(train_x)
        if verbose:
            print("predicting validation set")
        predicted_val_y = classifier.predict(test_x)

        train_accuracy = (predicted_train_y == train_y).astype(np.float32).mean() * 100
        val_accuracy = (predicted_val_y == test_y).astype(np.float32).mean() * 100

        if verbose:
            print(f"classification accuracy: test={val_accuracy}%, train={train_accuracy}%")

        return {
            "ratio": compression_ratio,
            "train_accuracy": float(train_accuracy),
            "val_accuracy": float(val_accuracy),
            "train_time": train_time,
            "throughput": throughput,
        }


    def run(self):
        try:
            params_list = list(self.iter_model_params())
            for params in tqdm(params_list):

                for i in range(self.num_random_trails):
                    id = self._db.to_id({**params, "trial": i})
                    if self._db.has(id):
                        continue

                    model = ConvModel(**params)
                    result = self.test_classification(model)

                    with io.BytesIO() as fp:
                        torch.save(model, fp)
                        fp.seek(0)
                        self._db.store(id, data=fp.read(), meta={"params": params, "result": result})
        except KeyboardInterrupt:
            pass
        return self

    def get_dataframe(self) -> pd.DataFrame:
        param_map = {}
        for id, data, meta in self._db.iter():
            param_id = self._db.to_id(meta["params"])
            param_map.setdefault(param_id, []).append(meta)

        rows = []
        for trials in param_map.values():
            row = {
                **{
                    key: ", ".join(map(str, value))
                    for key, value in trials[0]["params"].items()
                },
                **{
                    key: sum(t["result"][key] for t in trials) / len(trials)
                    for key in trials[0]["result"].keys()
                }
            }
            rows.append(row)

        return pd.DataFrame(rows).sort_values("val_accuracy")

    def dump(self):
        df = self.get_dataframe()
        print(df.to_markdown())


if __name__ == "__main__":
    (ConvModelTester()
        .run()
        .dump()
    )
