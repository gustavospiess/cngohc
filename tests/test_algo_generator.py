'''
Tests for `hogc.algo.generator` module.
'''

from hogc.algo import generator
from hogc import models
from collections import Counter
from functools import reduce

import pytest


def test_generate_zero():
    a = generator.make_vertex([0])
    assert a[0] == 0


def test_generate_one():
    count = Counter([int(generator.make_vertex([1])[0]) for _ in range(1000)])
    assert 1000 * 0.7 > count[0] > 1000 * 0.6
    assert 1000 * 0.3 > count[1] + count[-1] > 1000 * 0.2
    count = Counter([int(generator.make_vertex([10])[0]) for _ in range(1000)])
    assert 1000 * 0.12 > count[0] > 1000 * 0.06


@pytest.mark.parametrize('data,length', [
    ((1,), 1),
    (range(2), 2),
    ([1, 2, 3], 3),
    ([0.1, 1.0, 10.0, 12.2, 0.1234], 5),
    ])
def test_generate_len(data, length):
    vtx = generator.make_vertex(data)
    assert isinstance(vtx, models.Vertex)
    assert len(vtx) == length


@pytest.mark.parametrize('qtd', [
    1,
    4,
    25,
    100,
    1000
    ])
def test_initialize_graph(qtd):
    p = generator.Parameters(vertex_count=qtd)
    g = generator.initialize_graph(p)
    assert len(g.vertex_set) == qtd


@pytest.mark.parametrize('qtd', [
    1,
    4,
    25,
    100,
    1000
    ])
@pytest.mark.parametrize('data,length', [
    ((1,), 1),
    (range(2), 2),
    ([1, 2, 3], 3),
    ([0.1, 1.0, 10.0, 12.2, 0.1234], 5),
    ])
def test_initialize_graph_vertex_len(qtd, data, length):
    p = generator.Parameters(vertex_count=qtd, deviation_sequence=data)
    g = generator.initialize_graph(p)
    assert len(g.vertex_set) == qtd
    for v in g.vertex_set:
        assert len(v) == length


@pytest.mark.parametrize('vertex_count', [
    100,
    500,
    1000,
    ])
@pytest.mark.parametrize('deviation_sequence', [
    (10,),
    (1, 2,),
    (1, 2, 3,),
    (1, 2, 3,) * 3,
    ])
@pytest.mark.parametrize('community_count', [
    (2, 3,),
    (3, 2,),
    (2, 2, 2,),
    ])
def test_initialize_communities(
        vertex_count,
        deviation_sequence,
        community_count):
    p = generator.Parameters(
            vertex_count=vertex_count,
            deviation_sequence=deviation_sequence,
            community_count=community_count)
    g = generator.initialize_graph(p)
    g = generator.initialize_communities(p, g)

    assert len(g.partition) == community_count[0]

    def partition_depht(part, n=0):
        yield part, n
        for sub in part:
            if isinstance(sub, models.Partition):
                yield from partition_depht(sub, n+1)

    for expected_level in range(1, len(community_count)+1):
        expected = reduce(
                lambda x, y: x*y,
                community_count[:expected_level],
                1)
        communities_of_level = tuple(
                part for part, level in partition_depht(g.partition)
                if level == expected_level)
        assert len(communities_of_level) == expected

    proccessed = tuple(g.partition.depht)
    for vertex in proccessed:
        neighbor_list = tuple(g.neighbors_of[vertex])
        assert len(neighbor_list) > 0
        for neighbor in neighbor_list:
            assert neighbor in proccessed

    for vertex in proccessed:
        assert g.partition in g.partitions_of[vertex]
        assert len(set(g.partition) & set(g.partitions_of[vertex])) == 1

    for vertex in proccessed:
        print(1)
        print(tuple(g.partitions_of[vertex]))
        for part in g.partitions_of[vertex]:
            print(part)
            assert vertex in part.representative_set


def test_batch_generator():
    p = generator.Parameters(
            vertex_count=1000,
            deviation_sequence=(1.0, 3.0),
            community_count=(3,),
            homogeneity_indicator=0.0)
    g = generator.initialize_graph(p)
    for b in generator.batch_generator(g):
        assert 0 < len(b) < 1000
    assert sum(len(b) for b in generator.batch_generator(g)) == 1000


def test_chose_partition():
    p = generator.Parameters(
            vertex_count=1000,
            deviation_sequence=(3.0, 3.0),
            community_count=(3,),
            homogeneity_indicator=0.0)
    g = generator.initialize_graph(p)

    partition_edge_set = set()
    v1 = max(g.vertex_set, key=lambda v: v[0])
    v2 = min(g.vertex_set, key=lambda v: v[0])
    v3 = max(g.vertex_set, key=lambda v: v[1])
    v4 = min(g.vertex_set, key=lambda v: v[1])

    p1 = models.WeighedPartition(
            v1, weigh_vector=models.Vector((1, 0)), representative_set={v1})
    p2 = models.WeighedPartition(
            v2, weigh_vector=models.Vector((1, 0)), representative_set={v2})
    p3 = models.WeighedPartition(
            v3, weigh_vector=models.Vector((0, 1)), representative_set={v3})
    p4 = models.WeighedPartition(
            v4, weigh_vector=models.Vector((0, 1)), representative_set={v4})

    partition = models.WeighedPartition(
            {p1, p2, p3, p4},
            weigh_vector=models.Vector((0.5, 0.5)),
            representative_set={models.Vertex((9999999, 99999999))})

    vt1 = models.Vertex((v1[0]+0.1, 1))
    vt2 = models.Vertex((v2[0]+0.1, 1))
    vt3 = models.Vertex((1, v3[1]+0.1))
    vt4 = models.Vertex((1, v4[1]+0.1))

    g = models.Graph(
            frozenset(g.vertex_set | {vt1, vt2, vt3, vt4}),
            frozenset(g.edge_set | partition_edge_set),
            partition
            )

    assert generator.chose_partition(p, g, vt1) == p1
    assert generator.chose_partition(p, g, vt2) == p2
    assert generator.chose_partition(p, g, vt3) == p3
    assert generator.chose_partition(p, g, vt4) == p4

    p = generator.Parameters(
            vertex_count=1000,
            deviation_sequence=(3.0, 3.0),
            community_count=(3,),
            homogeneity_indicator=1.0)
    count = Counter(generator.chose_partition(p, g, vt1) for _ in range(2500))
    assert count[p1] > 450
    assert count[p2] > 450
    assert count[p3] > 450
    assert count[p4] > 450
