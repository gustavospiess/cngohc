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
