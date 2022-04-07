'''
Tests for `hogc.algo.generator` module.
'''

from hogc.algo import generator
from hogc import models
from collections import Counter

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
