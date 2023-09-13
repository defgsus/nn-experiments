from typing import Optional, Union, Callable

import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise


class AutoCluster(ClusterMixin, BaseEstimator):

    SUPPORTED_DISTANCE_METRICS = [
        "euclidean", "l2", "l1", "manhattan", "cityblock",
        "braycurtis", "canberra", "chebyshev", "correlation",
        "cosine", "dice", "hamming", "jaccard", "kulsinski",
        "matching", "minkowski", "rogerstanimoto",
        "russellrao", "sokalmichener",
        "sokalsneath", "sqeuclidean", "yule",
        "nan_euclidean",
    ]

    def __init__(
            self,
            distance_threshold: float = 1.,
            distance_metric: Union[str, Callable] = "euclidean",
            max_n_clusters: int = 10,
            label_dtype: Union[str, np.dtype] = "int64",
            # random_state=None,
            verbose: bool = False,
    ):
        self.distance_threshold = distance_threshold
        self.distance_metric = distance_metric
        self.max_n_clusters = max_n_clusters
        self.label_dtype = label_dtype
        # self.random_state = random_state
        self.verbose = verbose
        self.cluster_centers_: Optional[np.ndarray] = None
        self.cluster_counts_: Optional[np.ndarray] = None
    
    @property
    def n_clusters_(self) -> int:
        if self.cluster_centers_ is None:
            return 0
        return self.cluster_centers_.shape[0]

    def fit(self, X: np.ndarray, y=None) -> "AutoCluster":
        self.cluster_centers_ = None
        self.cluster_counts_ = None

        self.partial_fit(X, y)
        return self

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def partial_fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.partial_fit(X, y, return_labels=True)

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def partial_fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.partial_fit(X, y, return_labels=True)

    def partial_fit(
            self,
            X: np.ndarray,
            y=None,
            counts: Optional[np.ndarray] = None,
            return_labels: bool = False,
    ) -> Union["AutoCluster", np.ndarray]:
        X = self._validate_data(
            X, y,
            dtype=[np.float64, np.float32, np.bool_],
            reset=self.cluster_centers_ is None,
        )
        if counts is not None:
            if counts.ndim != 1:
                raise ValueError(f"`counts` must have 1 dimension, got {counts.ndim}")
            if counts.shape[0] != X.shape[0]:
                raise ValueError(f"Expected `counts` to be of length {X.shape[0]}, got {counts.shape[0]}")

        # random_state = check_random_state(self.random_state)
        if return_labels:
            labels = np.ndarray((X.shape[0], ), dtype=self.label_dtype)

        label_idx = 0
        while X.shape[0]:
            if self.cluster_centers_ is None:
                # create first cluster
                if counts is None:
                    count = 1
                else:
                    count = counts[0]
                    counts = counts[1:]
                self.cluster_centers_ = np.copy(X[0]).reshape(1, -1)
                self.cluster_counts_ = np.array([count], dtype=self.label_dtype)
                X = X[1:]
                if return_labels:
                    labels[label_idx] = 0
                    label_idx += 1
                continue

            distances = self._distance_metric(X)
            best_ids = np.argmin(distances, axis=-1)
            best_distances = np.min(distances, axis=-1)

            for idx, (feature, cluster_id, distance) in enumerate(zip(X, best_ids, best_distances)):
                if distance > self.distance_threshold and self.n_clusters_ < self.max_n_clusters:
                    # create a new cluster
                    cluster_id = self.n_clusters_
                    self.cluster_centers_ = np.resize(
                        self.cluster_centers_, (cluster_id + 1, self.cluster_centers_.shape[1])
                    )
                    self.cluster_counts_ = np.resize(self.cluster_counts_, (cluster_id + 1, ))
                    if counts is None:
                        count = 1
                    else:
                        count = counts[0]
                        counts = counts[idx + 1:]
                    self.cluster_centers_[cluster_id] = feature
                    self.cluster_counts_[cluster_id] = count
                    X = X[idx + 1:]
                    if return_labels:
                        labels[label_idx] = cluster_id
                        label_idx += 1
                    # start again with new distance matrix
                    break

                if counts is None:
                    count = 1
                else:
                    count = counts[idx]

                self._add_to_cluster(cluster_id, feature, count)
                if return_labels:
                    labels[label_idx] = cluster_id
                    label_idx += 1

            if idx == X.shape[0] - 1:
                break

        if return_labels:
            return labels
        else:
            return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_data(
            X,
            dtype=[np.float64, np.float32, np.bool_],
            reset=False,
        )
        distances = self._distance_metric(X)
        best_ids = np.argmin(distances, axis=-1)
        return best_ids

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)

    def _add_to_cluster(self, cluster_id: int, feature: np.ndarray, count: int):
        cur_count = self.cluster_counts_[cluster_id]

        if self.cluster_centers_.dtype == np.bool_:
            self.cluster_centers_[cluster_id] = (
                ((
                    self.cluster_centers_[cluster_id].astype("float64") * cur_count
                    + feature.astype("float64") * count
                ) / (cur_count + count)).astype("bool")
            )
        else:
            self.cluster_centers_[cluster_id] = (
                (self.cluster_centers_[cluster_id] * cur_count + feature * count) / (cur_count + count)
            )
        self.cluster_counts_[cluster_id] = cur_count + count

    def _distance_metric(self, X: np.ndarray):
        return pairwise.pairwise_distances(
            X,
            self.cluster_centers_,
            metric=self.distance_metric,
        )

    def reduce_clusters(self, below: int) -> "AutoCluster":
        """
        Remove clusters whose count is below `below`
        :param below: int
        :return: self
        """
        if self.cluster_counts_ is None:
            return self

        remove_index = self.cluster_counts_ < below
        keep_index = np.invert(remove_index)

        X = self.cluster_centers_[remove_index]
        counts = self.cluster_counts_[remove_index]

        self.cluster_centers_ = self.cluster_centers_[keep_index]
        self.cluster_counts_ = self.cluster_counts_[keep_index]

        prev_max_n_clusters = self.max_n_clusters
        self.max_n_clusters = max(1, self.n_clusters_)

        self.partial_fit(X, counts=counts)

        self.max_n_clusters = prev_max_n_clusters
        return self
