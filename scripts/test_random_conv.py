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
from src.util import iter_parameter_permutations
from src.util.binarydb import BinaryDB


class ConvModel(nn.Module):

    def __init__(
            self,
            channels: List[int],
            kernel_size: List[int],
            stride: List[int],
            dilation: List[int],
            activation: List[Optional[str]],
            input_channels: int = 3,
    ):
        super().__init__()

        self.layers = nn.Sequential()
        for i in range(len(channels)):
            self.layers.append(
                nn.Conv2d(
                    in_channels=input_channels if i == 0 else channels[i - 1],
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
            model_count: Optional[int] = None,
            layers_range: Tuple[int, int] = (3, 3),
            channel_choices: Tuple[int, ...] = (32, ), #32, 48, 64),
            kernel_size_choices: Tuple[int, ...] = (3, 5, 7, 9),
            stride_choices: Tuple[int, ...] = (1, 2, 3),
            dilation_choices: Tuple[int, ...] = (1, 2, 3),
            activation_choices: Tuple[Optional[str], ...] = ("relu", ),
            input_shape: Tuple[int, int, int] = (3, 96, 96),
            ratio_range: Optional[Tuple[float, float]] = (0., 1.), #(.28, .9),
            seed: int = 667,
            dataset_max_size_train: int = 500,
            dataset_max_size_test: int = 500,
            num_random_trails: int = 5,
            min_val_accuracy: Optional[float] = 29.,
            min_throughput: Optional[float] = 1000.,
            do_store_models: bool = False,
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
        self.min_val_accuracy = min_val_accuracy
        self.min_throughput = min_throughput
        self.do_store_models = do_store_models
        self.cache_path = cache_path
        self._dataset = None
        self._db = BinaryDB(self.cache_path / "db-ch32.sqlite")
        print(f"num permutations: {self.num_permutations():,}")

    def num_permutations(self):
        num = 1
        for num_layers in range(self.layers_range[0], self.layers_range[1] + 1):
            for choices in (
                    self.channel_choices,
                    self.kernel_size_choices,
                    self.stride_choices,
                    self.dilation_choices,
                    self.activation_choices,
            ):
                num *= len(choices) ** num_layers
        return num

    @torch.no_grad()
    def iter_model_params(self) -> Generator[dict, None, None]:
        params_hash_set = set()
        self._num_duplicates = 0
        self._num_skipped = 0
        self._num_yielded = 0
        if self.model_count is None:
            iterable = self._iter_all_model_params()
        else:
            iterable = self._iter_random_model_params()

        # iterable = shuffle_iter(iterable)

        for num_tried, params in enumerate(iterable):
            if self.model_count is not None and self._num_yielded >= self.model_count:
                break

            params_hash = str(params)
            if params_hash in params_hash_set:
                self._num_duplicates += 1
                continue
            params_hash_set.add(params_hash)

            model = ConvModel(**params)

            try:
                output_shape = model(torch.ones(*self.input_shape)).shape
            except RuntimeError:
                self._num_skipped += 1
                continue

            if self.ratio_range is not None:
                ratio = math.prod(output_shape) / math.prod(self.input_shape)
                if not self.ratio_range[0] <= ratio <= self.ratio_range[1]:
                    self._num_skipped += 1
                    continue

            yield params
            self._num_yielded += 1

    def _iter_random_model_params(self):
        for num_tied in range(self.num_permutations() * 10):
            if self._num_yielded >= self.model_count:
                break

            num_layers = self.rng.randint(*self.layers_range)
            yield {
                "channels": [self.rng.choice(self.channel_choices) for _ in range(num_layers)],
                "kernel_size": [self.rng.choice(self.kernel_size_choices) for _ in range(num_layers)],
                "stride": [self.rng.choice(self.stride_choices) for _ in range(num_layers)],
                "dilation": [self.rng.choice(self.dilation_choices) for _ in range(num_layers)],
                "activation": [self.rng.choice(self.activation_choices) for _ in range(num_layers)],
            }

    def _iter_all_model_params(self):
        for num_layers in range(self.layers_range[0], self.layers_range[1] + 1):
            matrix = {}
            for i in range(num_layers):
                matrix[f"activation-{i}"] = self.activation_choices
                matrix[f"dilation-{i}"] = self.dilation_choices
                matrix[f"stride-{i}"] = self.stride_choices
                matrix[f"kernel_size-{i}"] = self.kernel_size_choices
                matrix[f"channels-{i}"] = self.channel_choices

            for param in iter_parameter_permutations(matrix):
                yield {
                    "channels": [param[f"channels-{i}"] for i in range(num_layers)],
                    "kernel_size": [param[f"kernel_size-{i}"] for i in range(num_layers)],
                    "stride": [param[f"stride-{i}"] for i in range(num_layers)],
                    "dilation": [param[f"dilation-{i}"] for i in range(num_layers)],
                    "activation": [param[f"activation-{i}"] for i in range(num_layers)],
                }

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

        def _result_meets_criteria(result: dict):
            if self.min_val_accuracy is not None and result["val_accuracy"] < self.min_val_accuracy:
                return False

            if self.min_throughput is not None and result["throughput"] < self.min_throughput:
                return False

            return True

        try:
            with tqdm(total=self.num_permutations() if self.model_count is None else self.model_count) as progress:
                num_skipped = 0
                num_existing = 0
                fully_tested = 0
                for params in self.iter_model_params():
                    progress.set_postfix({
                        "duplicates": self._num_duplicates,
                        "already in db": num_existing,
                        "skipped early": self._num_skipped,
                        "skipped late": num_skipped,
                        "fully_tested": fully_tested,
                    })
                    progress.update()

                    for i in range(self.num_random_trails):
                        id = self._db.to_id({**params, "trial": i})
                        if existing := self._db.get(id):
                            result = existing[1]["result"]
                            num_existing += 1
                            if not _result_meets_criteria(result):
                                break

                            continue

                        model = ConvModel(**params)
                        result = self.test_classification(model)

                        data = None
                        if self.do_store_models:
                            with io.BytesIO() as fp:
                                torch.save(model, fp)
                                fp.seek(0)
                                data = fp.read()

                        self._db.store(id, data=data, meta={"params": params, "result": result})

                        if not _result_meets_criteria(result):
                            num_skipped += 1
                            break

                        if i == self.num_random_trails - 1:
                            fully_tested += 1

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
                },
                "min_val_acc": min(t["result"]["val_accuracy"] for t in trials),
                "max_val_acc": max(t["result"]["val_accuracy"] for t in trials),
                "trails": len(trials),
            }
            rows.append(row)

        return pd.DataFrame(rows).sort_values("val_accuracy")

    def dump(self):
        df = self.get_dataframe()
        print(df.to_markdown())


def shuffle_iter(iterable, max_shuffle: int = 10_000) -> Generator:
    buffer = []
    for item in iterable:
        buffer.append(item)
        if len(buffer) >= max_shuffle:
            idx = random.randrange(len(buffer))
            yield buffer.pop(idx)

    while buffer:
        idx = random.randrange(len(buffer))
        yield buffer.pop(idx)


if __name__ == "__main__":
    (ConvModelTester()
        .run()
        #.dump()
    )
