'''
Tests aaaaaaafor `cngohc.algo.generator` module.
'''

from cngohc.algo import generator
from cngohc import models
from collections import Counter
from functools import reduce
from random import sample

import pytest
import typing as tp


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
    for part in g.partition.flat:
        for vertex in part.representative_set:
            assert g.degree_of[vertex] > 0

    for expected_level in range(1, len(community_count)+1):
        expected = reduce(
                lambda x, y: x*y,
                community_count[:expected_level],
                1)
        communities_of_level = tuple(
                part for part in g.partition.flat
                if part.level == expected_level)
        print(expected_level)
        assert len(communities_of_level) == expected

    proccessed = tuple(g.partition.depht)
    for vertex in proccessed:
        neighbor_list = tuple(g.neighbors_of[vertex])
        assert len(neighbor_list) > 0
        for neighbor in neighbor_list:
            assert neighbor in proccessed

    for vertex in proccessed:
        for part in g.partitions_of[vertex]:
            assert vertex in part.representative_set


def test_batch_generator():
    p = generator.Parameters(
            vertex_count=6000,
            deviation_sequence=(1.0, 3.0),
            community_count=(3,),
            homogeneity_indicator=0.0)
    g = generator.initialize_graph(p)
    for b in generator.batch_generator(g):
        assert 1 <= len(b) <= 2048
    assert sum(len(b) for b in generator.batch_generator(g)) == 6000


def test_edge_insertion_within():
    p = generator.Parameters(
            deviation_sequence=(10,),
            )
    g = generator.initialize_graph(p)
    g = generator.initialize_communities(p, g)
    for part in sample(g.partition.flat, k=3):
        for v in sample(g.vertex_set, k=10):
            edge_set = generator.edge_insertion_within(p, g, v, part, True)
            assert len(edge_set) > 0
            for edge in edge_set:
                assert v in edge
                other = edge[0] if v == edge[1] else edge[1]
                assert other in part


def test_edge_insertion_between():
    p = generator.Parameters(
            deviation_sequence=(10, 10,),
            )
    g = generator.initialize_graph(p)
    g = generator.initialize_communities(p, g)

    unprocessed = {v for v in g.vertex_set if g.degree_of[v] == 0}

    for part in sample(g.partition.flat, k=3):
        ignor = set(part.flat)
        for v in sample(unprocessed, k=30):
            limit = len(generator.edge_insertion_within(p, g, v, part, False))
            edge_set = generator.edge_insertion_between(p, g, v, ignor, limit)
            assert limit >= len(edge_set) >= 0
            for edge in edge_set:
                assert v in edge
                other = edge[0] if v == edge[1] else edge[1]
                assert len(tuple(g.neighbors_of[other])) > 0


def find_connected_components(g: models.Graph):
    '''Move to validations'''
    forest: tp.Set[tp.FrozenSet[models.Vertex]] = set()
    for edge in g.edge_set:
        a_set: tp.FrozenSet[models.Vertex] = frozenset({edge[0]})
        b_set: tp.FrozenSet[models.Vertex] = frozenset({edge[1]})
        for comp in forest:
            if edge[0] in comp:
                a_set = comp
            if edge[1] in comp:
                b_set = comp
        if a_set != b_set:
            if a_set in forest:
                forest.remove(a_set)
            if b_set in forest:
                forest.remove(b_set)
            forest.add(a_set | b_set)
        return forest


@pytest.mark.slow
def test_generator():
    p = generator.Parameters(
            vertex_count=1_500,
            min_edge_count=4_000,
            deviation_sequence=(1.0, 2.5),
            homogeneity_indicator=0.95,
            representative_count=10,
            community_count=(2, 3, 4),
            max_within_edge=(1, 5, 10, 30),
            max_between_edge=20)
    g = generator.generator(p)
    assert len(g.vertex_set) == p.vertex_count
    assert len(set(g.zero_degree_vertex)) == 0
    assert len(find_connected_components(g)) == 1
    for vertex in sample(g.vertex_set, k=100):
        assert len(tuple(g.partitions_of[vertex])) > 0
    assert len(g.edge_set) >= p.min_edge_count


@pytest.mark.slow
def test_generator_b():
    p = generator.Parameters(
            vertex_count=10_500,
            min_edge_count=4_000,
            deviation_sequence=(1.0, 2.5),
            homogeneity_indicator=0.95,
            representative_count=10,
            community_count=(4,),
            max_within_edge=(1, 30),
            max_between_edge=20)
    g = generator.generator(p)
    assert len(g.vertex_set) == p.vertex_count
    assert len(set(g.zero_degree_vertex)) == 0
    for vertex in sample(g.vertex_set, k=100):
        assert len(tuple(g.partitions_of[vertex])) > 0
    assert len(g.edge_set) >= p.min_edge_count
