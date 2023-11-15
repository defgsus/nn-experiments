from typing import Union, Optional, Callable, List, Tuple, Generator

import torch
from torch.utils.data import Dataset, IterableDataset

from src.util.embedding import normalize_embedding


class DissimilarImageIterableDataset(IterableDataset):

    def __init__(
            self,
            dataset: Union[IterableDataset, Dataset],
            max_similarity: float = .9,
            max_age: Optional[int] = None,
            encoder: Union[str, torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]] = "flatten",
            batch_size: int = 10,
            yield_bool: bool = False,
            verbose: bool = False,
    ):
        if isinstance(encoder, str):
            assert encoder in ("flatten", "flatten-norm"), f"Got '{encoder}'"

        self.dataset = dataset
        self.max_similarity = float(max_similarity)
        self.max_age = max_age
        self.encoder = encoder
        self.batch_size = int(batch_size)
        self.yield_bool = yield_bool
        self.verbose = bool(verbose)
        self.features: Optional[torch.Tensor] = None
        self._ages: Optional[List[int]] = None
        self._age = 0

    def __iter__(self) -> Generator[Union[torch.Tensor, Tuple[torch.Tensor, ...]], None, None]:
        self.features = None
        self._ages = None
        self._age = 0
        self._num_passed = 0
        image_batch = []
        tuple_batch = []

        def _process(data):

            is_tuple = isinstance(data, (tuple, list))
            if is_tuple:
                image_batch.append(data[0])
                tuple_batch.append(data[1:])
            else:
                image_batch.append(data)
                tuple_batch.append(None)

            if len(image_batch) >= self.batch_size:
                yield from self._process_batch(image_batch, tuple_batch)
                image_batch.clear()
                tuple_batch.clear()

            self._age += 1

            self._drop_old_features()

        if not self.verbose:
            for data in self.dataset:
                yield from _process(data)

        else:
            from tqdm import tqdm

            try:
                total = len(self.dataset)
            except:
                total = None

            with tqdm(total=total) as progress:
                for data in self.dataset:
                    yield from _process(data)

                    progress.desc = (
                        f"filtering dissimilar images"
                        f" (features={self.features.shape[0] if self.features is not None else 0}"
                        f", passed={self._num_passed})"
                    )
                    progress.update(1)

        if image_batch:
            yield from self._process_batch(image_batch, tuple_batch)

    def _process_batch(self, image_batch, tuple_batch):
        image_batch = torch.concat([i.unsqueeze(0) for i in image_batch])
        feature_batch = self._encode(image_batch)

        # store first image feature
        if self.features is None:
            if not self.yield_bool:
                if tuple_batch[0]:
                    yield image_batch[0], *tuple_batch[0]
                else:
                    yield image_batch[0]
            else:
                if tuple_batch[0]:
                    yield image_batch[0], True, *tuple_batch[0]
                else:
                    yield image_batch[0], True

            self._num_passed += 1
            self.features = feature_batch[0].unsqueeze(0)
            self._ages = [self._age]
            image_batch = image_batch[1:]
            tuple_batch = tuple_batch[1:]
            feature_batch = feature_batch[1:]

        similarities = self._highest_similarities(feature_batch)

        features_to_add = None
        for image, tup, feature, similarity in zip(image_batch, tuple_batch, feature_batch, similarities):

            if features_to_add is not None:
                # get highest similarity with stored features and new features from batch
                similarity2 = self._highest_similarities(feature.unsqueeze(0), features_to_add)
                similarity = torch.max(similarity, similarity2)

            does_pass = similarity <= self.max_similarity

            # always yield when required
            if self.yield_bool:
                if tup:
                    yield image, does_pass, *tup
                else:
                    yield image, does_pass

            if does_pass:
                if not self.yield_bool:
                    if tup:
                        yield image, *tup
                    else:
                        yield image

                self._num_passed += 1
                self._ages.append(self._age)
                if features_to_add is None:
                    features_to_add = feature.unsqueeze(0)
                else:
                    features_to_add = torch.concat([features_to_add, feature.unsqueeze(0)])

        if features_to_add is not None:
            self.features = torch.concat([self.features, features_to_add])

    def _drop_old_features(self):
        if self.max_age is not None and self._ages:
            idx = None
            for i, age in enumerate(self._ages):
                if self._age - age > self.max_age:
                    idx = i + 1
                else:
                    break

            if idx is not None and idx < len(self._ages):
                self.features = self.features[idx:]
                self._ages = self._ages[idx:]

    def _highest_similarities(self, feature_batch: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        sim = feature_batch @ (features if features is not None else self.features).T
        return sim.max(dim=1)[0]

    def _encode(self, image_batch: torch.Tensor) -> torch.Tensor:
        if isinstance(self.encoder, str):

            if self.encoder == "flatten":
                feature_batch = image_batch.flatten(1)

            elif self.encoder == "flatten-norm":
                feature_batch = normalize_embedding(image_batch.flatten(1))

            elif self.encoder.startswith("clip"):
                from src.models.clip import ClipSingleton
                feature_batch = ClipSingleton.encode_image(image_batch)

            elif self.encoder.startswith("encoderconv:"):
                from src.models.encoder import EncoderConv2d
                if not hasattr(self, "_encoderconv"):
                    self._encoderconv = EncoderConv2d.from_torch(self.encoder.split(":", 1)[-1])
                feature_batch = self._encoderconv.encode_image(image_batch)

            else:
                raise ValueError(f"Unsupported encoder '{self.encoder}', expected 'flatten', 'clip'")

        elif callable(self.encoder):
            feature_batch = self.encoder(image_batch)
        else:
            raise ValueError(f"Unsupported encoder type {type(self.encoder).__name__}, expected str or callable")

        # feature_batch = feature_batch - feature_batch.mean(dim=1, keepdim=True)
        return feature_batch / torch.norm(feature_batch, dim=1, keepdim=True)
