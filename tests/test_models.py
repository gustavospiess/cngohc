'''Tests for `hogc.models` module.'''


from hogc import models
import json
import tempfile
import pytest


def test_factorty():
    v = models.Vertex((0, 1,))
    p = models.Partition((v,))
    g = models.Graph(
            frozenset((v,)),
            frozenset(),
            p
            )
    assert v in p
    assert v in g.partition
    assert v in g.vertex_set


def test_partition_hash():
    p = models.Partition()
    p2 = models.Partition(identifier=p.identifier)
    p3 = models.Partition()
    assert p == p2
    assert hash(p) == hash(p2)
    assert p == p3
    assert hash(p3) != hash(p2)


def test_partition_contains():
    v = models.Vertex((1,))
    p2 = models.Partition((v,))
    p = models.Partition((p2,))
    assert 1 not in p
    assert v in p
    assert v in p2
    v2 = models.Vertex((2,))
    assert v2 not in p


def test_partition_json():
    data = models.Vertex((1, 2,))
    p2 = models.Partition((data,))
    p = models.Partition((p2,))

    serial = json.dumps(p.to_raw())
    raw = json.loads(serial)
    q = models.Partition.from_raw(raw)
    assert q == p
    assert hash(p) == hash(q)
    assert p.identifier == q.identifier


def test_weighedpartition_json():
    data = models.Vertex((1, 2,))
    p = models.Partition((data,), weigh_vector=(0, 0))
    p2 = models.Partition((data,))

    assert p.to_raw() != p2.to_raw()

    serial = json.dumps(p.to_raw())
    raw = json.loads(serial)
    q = models.Partition.from_raw(raw)
    assert q == p
    assert hash(p) == hash(q)
    assert p.identifier == q.identifier
    assert p.weigh_vector == q.weigh_vector


def test_partition_invalid_raw():
    with pytest.raises(TypeError):
        models.Partition.from_raw(None)
    with pytest.raises(TypeError):
        models.Partition.from_raw({})
    with pytest.raises(TypeError):
        models.Partition.from_raw({'identifier': 0, 'members': None})
    with pytest.raises(TypeError):
        models.Partition.from_raw({'identifier': 0, 'members': [1]})
    with pytest.raises(TypeError):
        models.Partition.from_raw({'identifier': 0, 'members': [1]})


def test_graph_neighbors():
    v1 = models.Vertex((1.0, 2.0,))
    v2 = models.Vertex((2.0, 2.0,))
    v3 = models.Vertex((3.0, 2.0,))
    v4 = models.Vertex((4.0, 2.0,))
    edge_set = set()
    edge_set.add((v1, v2,))
    edge_set.add((v1, v3))
    edge_set.add((v1, v4,))
    edge_set.add((v2, v3,))
    edge_set.add((v2, v4,))
    edge_set.add((v3, v4,))
    vertex_set = frozenset((v1, v2, v3, v4,))
    g = models.Graph(vertex_set, frozenset(edge_set))

    assert len(tuple(g.neighbors_of)) == 4
    assert len(tuple(g.neighbors_of[v1])) == 3
    assert len(tuple(g.neighbors_of[v2])) == 3
    assert len(tuple(g.neighbors_of[v3])) == 3
    assert len(tuple(g.neighbors_of[v4])) == 3

    assert v1 not in g.neighbors_of[v1]
    assert v2 in g.neighbors_of[v1]
    assert v3 in g.neighbors_of[v1]
    assert v4 in g.neighbors_of[v1]

    assert v1 in g.neighbors_of[v2]
    assert v2 not in g.neighbors_of[v2]
    assert v3 in g.neighbors_of[v2]
    assert v4 in g.neighbors_of[v2]

    assert v1 in g.neighbors_of[v3]
    assert v2 in g.neighbors_of[v3]
    assert v3 not in g.neighbors_of[v3]
    assert v4 in g.neighbors_of[v3]

    assert v1 in g.neighbors_of[v4]
    assert v2 in g.neighbors_of[v4]
    assert v3 in g.neighbors_of[v4]
    assert v4 not in g.neighbors_of[v4]

    assert v1 in tuple(g.neighbors_of)
    assert v2 in tuple(g.neighbors_of)
    assert v3 in tuple(g.neighbors_of)
    assert v4 in tuple(g.neighbors_of)


def test_graph_save_partition():
    v1 = models.Vertex((1.0, 2.0,))
    v2 = models.Vertex((2.0, 2.0,))
    vertex_set = set()
    vertex_set.add(v1)
    vertex_set.add(v2)
    edge_set = set()
    edge_set.add((v1, v2,))
    p1 = models.Partition((v1, v2,))

    g = models.Graph(frozenset(vertex_set), frozenset(edge_set), p1)

    path = tempfile.gettempprefix()

    with open(path, 'w') as buf:
        g.write_partition_to_buffer(buf)
    with open(path, 'r') as buf:
        lines = buf.readlines()

    partition_raw = json.loads(lines[0])
    partition = models.Partition.from_raw(partition_raw)
    assert partition == g.partition
    assert hash(partition) == hash(g.partition)

    g_loaded = models.Graph()
    with open(path, 'r') as buf:
        g_loaded = g_loaded.read_partition_from_buffer(buf)
    assert g_loaded.partition == g.partition
    assert hash(g_loaded.partition) == hash(g.partition)


def test_graph_save_vertex():
    g = models.Graph()
    assert len(g.vertex_set) == 0

    v1 = models.Vertex((1.0, 2.0,))
    v2 = models.Vertex((2.0, 2.0,))
    g = models.Graph(frozenset((v1, v2)))

    assert len(g.vertex_set) == 2

    path = tempfile.gettempprefix()
    with open(path, 'w') as buf:
        g.write_vertex_to_buffer(buf)
    with open(path, 'r') as buf:
        lines = buf.readlines()

    assert lines[0] == '1.0,2.0\n'
    assert lines[1] == '2.0,2.0\n'

    g_loaded = models.Graph()
    with open(path, 'r') as buf:
        g_loaded = g_loaded.read_vertex_from_buffer(buf)
    assert g_loaded.vertex_set == g.vertex_set


def test_graph_save_edge():
    v1 = models.Vertex((1.0, 2.0,))
    v2 = models.Vertex((2.0, 2.0,))
    v3 = models.Vertex((3.0, 2.0,))
    v4 = models.Vertex((4.0, 2.0,))
    edge_set = set()
    edge_set.add((v1, v2,))
    edge_set.add((v2, v3,))
    edge_set.add((v3, v1,))
    edge_set.add((v1, v4,))
    edge_set.add((v2, v4,))
    edge_set.add((v3, v4,))

    g = models.Graph(edge_set=frozenset(edge_set))

    path = tempfile.gettempprefix()
    with open(path, 'w') as buf:
        g.write_edge_to_buffer(buf)
    with open(path, 'r') as buf:
        lines = buf.readlines()

    expected_lines = []
    for e in g.edge_set:
        expected_lines.append(f'{e[0][0]},{e[0][1]}\n')
        expected_lines.append(f'{e[1][0]},{e[1][1]}\n')

    for l1, l2 in zip(lines, expected_lines):
        assert l1 == l2

    g_loaded = models.Graph()
    with open(path, 'r') as buf:
        g_loaded = g_loaded.read_edge_from_buffer(buf)
    assert g_loaded.edge_set == g.edge_set


def test_graph_save():
    v1 = models.Vertex((1.0, 2.0,))
    v2 = models.Vertex((2.0, 2.0,))
    v3 = models.Vertex((3.0, 2.0,))
    v4 = models.Vertex((4.0, 2.0,))

    edge_set = set()
    edge_set.add((v1, v2,))
    edge_set.add((v2, v3,))
    edge_set.add((v3, v1,))
    edge_set.add((v1, v4,))
    edge_set.add((v2, v4,))
    edge_set.add((v3, v4,))

    p1 = models.Partition((v1, v2, v3, v4,))
    g = models.Graph(frozenset((v1, v2, v3, v4,)), frozenset(edge_set), p1)

    g_loaded = models.Graph()

    path = tempfile.gettempprefix()
    with open(path, 'w') as buf:
        g.write_vertex_to_buffer(buf)
    with open(path, 'r') as buf:
        g_loaded = g_loaded.read_vertex_from_buffer(buf)

    path = tempfile.gettempprefix()
    with open(path, 'w') as buf:
        g.write_edge_to_buffer(buf)
    with open(path, 'r') as buf:
        g_loaded = g_loaded.read_edge_from_buffer(buf)

    with open(path, 'w') as buf:
        g.write_partition_to_buffer(buf)
    with open(path, 'r') as buf:
        g_loaded = g_loaded.read_partition_from_buffer(buf)

    assert g_loaded.vertex_set == g.vertex_set
    assert g_loaded.edge_set == g.edge_set
    assert g_loaded.partition == g.partition
    assert hash(g_loaded.partition) == hash(g.partition)


def test_vertex_utils():
    v1 = models.Vertex((1.0, 2.0,))
    v2 = models.Vertex((2.0, 2.0,))
    v3 = models.Vertex((1.0, 0.0,))

    assert v2 - v1 == v3
    assert abs(v3) == 1
    assert abs(v1-v2) == 1
    assert v2 > v3

    assert ((v2-v1)+v1) == v2
    assert ((v2+v1)-v1) == v2

    p_1_1 = models.Vertex((1, 1))
    p_2_2 = models.Vertex((2, 2))
    p_3_3 = models.Vertex((3, 3))
    p_2_3 = models.Vertex((2, 3))

    assert (p_2_3 - p_3_3) < (p_2_2 - p_1_1)
    assert (p_3_3 - p_2_3) < (p_2_2 - p_1_1)
    assert (p_2_3 - p_3_3) < (p_1_1 - p_2_2)
    assert (p_3_3 - p_2_3) < (p_1_1 - p_2_2)

    s1, s2 = sorted((p_3_3, p_2_3))
    assert s1 == p_2_3
    assert s2 == p_3_3

    assert v1 == (1, 2,)
    assert v1 >= (1, 2,)
    assert v1 <= (1, 2,)
    assert not v1 < (1, 2,)
    assert not v1 > (1, 2,)


def test_iterate_partition_depth():
    v1 = models.Vertex((1, 1))
    v2 = models.Vertex((2, 2))
    p1 = models.Partition((v1, v2))
    v3 = models.Vertex((3, 3))
    p2 = models.Partition((v3, p1))

    assert v1 in tuple(p2.depht)
    assert v2 in tuple(p2.depht)
    assert v3 in tuple(p2.depht)

    assert p1 not in tuple(p2.depht)


def test_inverse_max_inertia_axis_base():
    v10 = models.Vertex((0, 10))
    v15 = models.Vertex((0, 15))
    v20 = models.Vertex((0, 25))

    p = models.Partition((v10, v15, v20))
    axis = p.inverse_max_inertia_axis()

    assert axis == models.Vertex((1, 0))


@pytest.mark.parametrize('noise', [
    (1000, 14),
    (10, 10),
    (1000000, 1),
    (100, 100),
    ])
def test_inverse_max_inertia_axis_noise(noise):
    v20 = models.Vertex((0, 5))
    v20_b = models.Vertex((0, 5)) + noise
    v10 = models.Vertex((0, 10))
    v15 = models.Vertex((0, 15))

    p = models.Partition((v10, v15, v20))
    p_b = models.Partition((v10, v15, v20_b))
    axis = p.inverse_max_inertia_axis()
    axis_b = p_b.inverse_max_inertia_axis()

    assert axis != axis_b
    assert round(sum(axis_b), 9) == 1
    assert round(sum(axis), 9) == 1

    centered = tuple(v - axis_b for v in p_b.depht)
    balanced_inertia = sum(sum(abs(v*a)**2 for v, a in zip(vec, axis_b))
                           for vec in zip(*centered))
    alt_axis = (0.5, 0.5)
    inertia = sum(sum(abs(v*a)**2 for v, a in zip(vec, alt_axis))
                  for vec in zip(*centered))

    assert balanced_inertia*0.75 < inertia


def test_inverse_max_inertia_axis_corner():
    v1 = models.Vertex((3, 0))
    v2 = models.Vertex((0, 3))
    v3 = models.Vertex((3, 3))

    p = models.Partition((v1, v2, v3))
    axis = p.inverse_max_inertia_axis()

    assert axis == models.Vertex((0.5, 0.5))


def test_inverse_max_inertia_axis_five_dimentions():
    v0 = models.Vertex((2, 1, 1, 1, 1))
    v1 = models.Vertex((1, 2, 1, 1, 1))
    v2 = models.Vertex((1, 1, 2, 1, 1))
    v3 = models.Vertex((1, 1, 1, 2, 1))
    v4 = models.Vertex((1, 1, 1, 1, 2))
    v5 = models.Vertex((0, 1, 1, 1, 1))
    v6 = models.Vertex((1, 0, 1, 1, 1))
    v7 = models.Vertex((1, 1, 0, 1, 1))
    v8 = models.Vertex((1, 1, 1, 0, 1))
    v9 = models.Vertex((1, 1, 1, 1, 0))

    p = models.Partition((v0, v1, v2, v3, v4, v5, v6, v7, v8, v9))
    axis = p.inverse_max_inertia_axis()

    assert tuple(map(lambda x: round(x, 9), axis)) == (0.2, 0.2, 0.2, 0.2, 0.2)


def test_compatibility_within_weigh():
    p = models.Partition(
            weigh_vector=models.Vector((1, 0)))

    assert p.weighed_distance(
            models.Vertex((1, 0)), models.Vertex((2, 0))) == 1
    assert p.weighed_distance(
            models.Vertex((1, 0)), models.Vertex((0, 1))) == 1
    assert p.weighed_distance(
            models.Vertex((1, 99)), models.Vertex((0, 1))) == 1

    p = models.Partition(
            weigh_vector=models.Vector((0.5, 0.5)))

    assert p.weighed_distance(
            models.Vertex((1, 0)), models.Vertex((3, 0))) == 2
    assert p.weighed_distance(
            models.Vertex((1, 0)), models.Vertex((0, 1))) == 1
    assert p.weighed_distance(
            models.Vertex((1, 99)), models.Vertex((0, 1))) > 100

    p = models.Partition(
            weigh_vector=models.Vector((0.7, 0.3)))

    assert p.weighed_distance(
            models.Vertex((1, 0)), models.Vertex((0, 0))) == 0.7
    assert p.weighed_distance(
            models.Vertex((0, 1)), models.Vertex((0, 0))) == 0.3
    assert p.weighed_distance(
            models.Vertex((1, 1)), models.Vertex((0, 0))) == 1


def test_graph_immutable():
    g1 = models.Graph()
    g2 = models.Graph()
    assert g1 == g2
