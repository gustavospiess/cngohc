'''Validation arlgorithm'''

from .models import Vector, Graph, Vertex


from pprint import pprint
from multiprocessing import Pool
from time import time
import networkx as nx


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


def relative_inertia(graph: Graph):
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


def modularity_part(data):
    global __graph
    global __partition_qt_of
    global __degree_of
    global __degree_sum
    global __neighbors

    community, pairs = data
    s = 0
    for vertex_a, vertex_b in pairs:
        degree_a = __degree_of[vertex_a]
        degree_b = __degree_of[vertex_b]
        adjacency = 1 if vertex_a in __neighbors[vertex_b] else 0
        s += (
                (adjacency - ((degree_a*degree_b)/__degree_sum))
                /
                (__partition_qt_of[vertex_a]*__partition_qt_of[vertex_b]))
    return s


__graph = None
__degree_of = None
__partition_qt_of = None
__degree_sum: int = 0


def shens_modularity(graph: Graph):
    '''TODO'''
    global __graph
    __graph = graph
    global __degree_of
    __degree_of = __graph.degree_of
    global __partition_qt_of
    __partition_qt_of = {
            p: len(graph.partitions_of[p])-1 for p in graph.partition.depht}
    global __degree_sum
    __degree_sum = 2*len(graph.edge_set)
    global __neighbors
    __neighbors = __graph.neighbors_of

    community_set = tuple(sorted(
            (p for p in graph.partition.flat),
            key=lambda p: p.level * -1))

    # edges_of_part = {p: len(graph.edges_of_part[p]) for p in graph.partition.flat}
    # for community in graph.partition.flat:
    #     print(f'{community.level=}')
    #     print(f'{edges_of_part[community]=}')
    #     print(f'{100*edges_of_part[community]/(len(community.depht)*(len(community.depht)-1)/2)=}')
    #     print(f'{len(community.depht)=}')

    mod = 0
    mod_b = 0
    for community in community_set:
        generator = (
                (community, tuple((vertex_a, vertex_b) for vertex_a in community.depht))
                for vertex_b in community.depht)
        with Pool() as p:
            _map = p.imap_unordered(modularity_part, generator)
            # _map = map(modularity_part, generator)
            _mod = sum(_map)/(__degree_sum)
            mod += _mod
            print(
                    round(_mod, 5),
                    community.level,
                    round(mod, 5))


def diameter(graph: Graph):
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(graph.edge_set)
    d = nx.diameter(nx_graph)
    print(d)
