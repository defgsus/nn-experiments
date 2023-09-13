import unittest
from typing import Iterable, List

import numpy as np
import sklearn.base

from src.cluster.autocluster import AutoCluster
from tests.base import TestBase


class TestAutoCluster(TestBase):

    def test_100_all_metrics(self):
        bool_metrics = [
            "dice", "jaccard", "kulsinski", "matching", "rogerstanimoto",
            "russellrao", "sokalmichener", "sokalsneath", "yule"
        ]
        for metric in AutoCluster.SUPPORTED_DISTANCE_METRICS:
            cluster = AutoCluster(distance_threshold=1, distance_metric=metric)

            rng = np.random.RandomState(110)
            if metric in bool_metrics:
                data = (rng.rand(1000, 10) < .5).astype("bool")
            else:
                data = rng.rand(1000, 10)

            cluster.partial_fit(data)
            # print(cluster.n_clusters_)

    def test_200_fit_reset(self):
        cluster = AutoCluster(distance_threshold=1.3)
        rng = np.random.RandomState(110)
        cluster.fit(rng.rand(100, 10))
        cluster.partial_fit(rng.rand(100, 10))
        self.assertEqual(200, np.sum(cluster.cluster_counts_))

        # can't partial_fit with different number of features
        with self.assertRaises(ValueError):
            cluster.partial_fit(rng.rand(100, 11))

        # but fit will reset the clusterer
        cluster.fit(rng.rand(10, 11))
        cluster.partial_fit(rng.rand(20, 11))
        self.assertEqual(30, np.sum(cluster.cluster_counts_))

        with self.assertRaises(ValueError):
            cluster.partial_fit(rng.rand(100, 10))

    def test_300_same_labels(self):
        """
        Check that incrementally assigning labels fits
        the final label transform.
        """
        for seed in range(20):
            seed = (seed + 7) * 13
            rng = np.random.RandomState(seed)

            cluster = AutoCluster(
                distance_threshold=1.,
                max_n_clusters=10,
                #distance_metric="correlation",
                distance_metric="euclidean",
            )

            features = rng.rand(1000, 10)
            labels1 = cluster.fit_transform(features)
            self.assertGreaterEqual(cluster.n_clusters_, 5)

            labels2 = cluster.transform(features)
            self.assertEqual(labels1.tolist(), labels2.tolist())

    def test_400_reduce(self):
        for seed in range(20):
            seed = (seed + 7) * 13
            rng = np.random.RandomState(seed)

            cluster = AutoCluster(
                distance_threshold=1.,
                max_n_clusters=30,
                distance_metric="euclidean",
            )

            features = rng.rand(100, 10)
            cluster.fit_transform(features)
            num_samples = cluster.cluster_counts_.sum()
            num_clusters = cluster.n_clusters_
            num_small_clusters = np.sum(cluster.cluster_counts_ <= 2)
            self.assertGreaterEqual(num_small_clusters, 3)

            cluster.reduce_clusters(below=3)
            self.assertEqual(num_samples, cluster.cluster_counts_.sum())
            self.assertEqual(num_clusters - num_small_clusters, cluster.n_clusters_)
