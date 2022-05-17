'''Validation arlgorithm'''

from .models import Vector, Graph, Vertex


from pprint import pprint


import typing as tp


def inertia(data: tp.Collection[Vector]) -> float:
    center = Vector.avg(data)
    return sum(v**2 for d in data for v in (d-center))


def find_connected_components(
        graph: Graph) -> tp.Set[tp.FrozenSet[Vertex]]:
    forest: tp.Set[tp.FrozenSet[Vertex]] = {frozenset([v]) for v in graph.vertex_set}
    for edge in graph.edge_set:
        a_set: tp.FrozenSet[Vertex] = frozenset({edge[0]})
        b_set: tp.FrozenSet[Vertex] = frozenset({edge[1]})
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
        if len(forest) == 1:
            break
    return forest


def connectivity(graph: Graph):
    '''TODO'''
    connected_comp = find_connected_components(graph)
    connected_componente_by_vertex = {
            vertex: comp
            for vertex in graph.vertex_set
            for comp in connected_comp
            if vertex in comp
            }
    for community in graph.partition.flat:
        comp_set = {
                connected_componente_by_vertex[vertex]
                for vertex in community.depht}
        assert len(comp_set) == 1


def relative_inertia(graph):
    '''TODO'''
    g_center = Vector.avg(graph.vertex_set)
    p_center = {p:Vector.avg(p.depht) for p in graph.partition.flat}

    inertia_by_level = {p.level: 0 for p in graph.partition.flat}
    qtd_by_level = {p.level: 0 for p in graph.partition.flat}
    for vertex in graph.vertex_set:
        for part in graph.partitions_of[vertex]:
            inertia = abs(vertex-p_center[part])**2
            inertia_by_level[part.level] += inertia
            qtd_by_level[part.level] += 1

    pprint(inertia_by_level)
    pprint(qtd_by_level)
    pprint({l: inertia_by_level[l]/qtd_by_level[l] for l in qtd_by_level})
