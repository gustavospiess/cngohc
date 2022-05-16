'''Tests for `cngohc.algo.clustering` module.'''


from cngohc.algo import clustering
from cngohc import models
import numpy as np

import pytest


@pytest.mark.parametrize('data', [
    ((1.4, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)),
    np.array([[1.4, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    ])
def test_k_medoids_protocol(data):
    k_medoids = clustering.KMedoids(data, n_clusters=2)

    assert isinstance(k_medoids[0], clustering.Cluster)
    assert isinstance(k_medoids[1], clustering.Cluster)

    assert isinstance(k_medoids, clustering.ClusterSet)

    assert set(tuple(p) for p in k_medoids.centroids) == {(10, 2), (1.4, 2)}
    assert all(node in data for node in k_medoids[0])
    assert all(node in data for node in k_medoids[1])
    assert (set(k_medoids[0]) & set(k_medoids[1])) == set()
    assert (set(k_medoids[0]) | set(k_medoids[1])) == {tuple(d) for d in data}
    assert len(k_medoids) == 2

    simple_data = tuple(tuple(d) for d in data)

    k1, k2 = tuple(sorted(k_medoids, key=lambda k: k[0][0]))
    assert tuple(k1.center) not in simple_data
    assert tuple(k2.center) in simple_data
    assert tuple(k1.centroid) in simple_data
    assert tuple(k2.centroid) in simple_data

    assert(len(k1) == 3)
    assert(len(k2) == 3)

    slc = k_medoids[:]
    assert slc[0] == k_medoids[0]
    assert slc[1] == k_medoids[1]


def test_k_medoids_len():
    data = ((1, 1), (4, 2), (0, 2), (1, 0), (10, 1), (10, 2), (10, 0))
    k_medoids = clustering.KMedoids(data, n_clusters=2)
    k1, k2 = tuple(sorted(k_medoids, key=lambda k: k[0][0]))
    assert(len(k1) == 4)
    assert(len(k2) == 3)

    assert k_medoids.min_len == 3

    caped_k_medoids = k_medoids.cap(3)

    ck1, ck2 = tuple(sorted(caped_k_medoids, key=lambda k: k[0][0]))
    assert(len(ck1) == 3)
    assert(len(ck2) == 3)

    assert set(v for v in ck1).issubset(set(v for v in k1))
    assert set(v for v in ck2) == (set(v for v in k2))

    assert set(v for v in ck1) == {(1, 1), (1, 0), (0, 2)}


def test_k_medois_vector():
    data = ((1, 2), (1.5, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0))
    k_medoids = clustering.KMedoids(data, n_clusters=2)

    assert isinstance(k_medoids, clustering.ClusterSet)
    assert isinstance(k_medoids[0], clustering.Cluster)
    assert isinstance(k_medoids[1], clustering.Cluster)
    assert isinstance(k_medoids[0].center, models.Vector)
    assert isinstance(k_medoids[1].center, models.Vector)
    assert isinstance(k_medoids.centers[0], models.Vector)
    assert isinstance(k_medoids.centers[1], models.Vector)
    assert all(isinstance(node, models.Vector) for node in k_medoids[0])
    assert all(isinstance(node, models.Vector) for node in k_medoids[0])
