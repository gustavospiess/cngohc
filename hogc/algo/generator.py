'''
This module wraps all the methods for the graph generation.
'''


from ..models import Vertex, Vector, Graph, Partition
from .rand import rand_norm, sample, rand_uni_range
from .clustering import KMedoids

import typing as tp


class Parameters(tp.NamedTuple):
    vertex_count: int = 1000  # N
    '''Number of vertexes of the graph'''
    min_edge_count: int = 300  # MTE
    '''Minimum number of edges of the graph'''
    deviation_sequence: tp.Tuple[float, ...] = (1, 2, 3,)  # A
    '''Sequence of deviation values to initialize the vertexes '''
    homogeneity_indicator: float = 0.9  # theta
    '''Ratio of vertexes to be added by homogeneity'''
    representative_count: int = 3  # NbRep
    '''Number of representatives of a partition'''
    community_count: tp.Tuple[int, ...] = (3, 2,)  # K
    '''
    Sequence of hierarchical communities quantities, the first value indicates
    how many communities will be created at the root of the graph, the second
    indicates how many will be created  inside each of the first ones, and so
    successively.

    The level_count, quantity of levels in the Graph, will be the length of it,
    and the amount of leaf communities will be the product of all those values.
    '''
    max_within_edge: tp.Tuple[int, ...] = (7, 12)  # E_max_wth
    '''
    Sequence of the max initial edges a vertex will receive when being added to
    a community, the first value is the quantity of edges to be added inside
    the first level community the vertex will be in, the second value for the
    second level community and so on.
    This should be a sequence of length equal to the level count of the graph.
    '''
    max_between_edge: int = 3  # E_max_btw
    '''Maximum quantity of initial edges a vertex will receive on addition to a
    community linking it to outside the community.'''


def make_vertex(deviation_seq: tp.Sequence[float]) -> Vertex:
    return Vertex(rand_norm(0, dev) for dev in deviation_seq)


def initialize_graph(param: Parameters) -> Graph:
    '''
    This initialization function takes the base parameters and returns a graph
    with populated with the vertex set.
    Each vertex is generated according to the `make_vertex` implementation.
    '''
    g = Graph(frozenset(
        make_vertex(param.deviation_sequence)
        for _ in range(param.vertex_count)))
    return g


def _initialize_communities(
        graph: Graph,
        param: Parameters,
        population: tp.FrozenSet[Vector] = frozenset(),
        level: int = 0
        ) -> tp.Tuple[tp.Set[tp.Tuple[Vertex, Vertex]], Partition]:
    '''
    Internal function for community initialization.

    It recursively runs a clustering method and initialize the communities.
    All the vertexes are in leaf communities.

    All the leaf communities are connected components of the graph.

    The return is the set of edges generates, as well as the partitions.
    '''

    if not population:
        return _initialize_communities(graph, param, graph.vertex_set, level)

    max_level = len(param.community_count)
    if level == max_level:
        return _initialize_leaf_communities(population, graph, param)
    comm_cont = param.community_count[level]

    sample_size = min(
            (param.representative_count ** (max_level-level)) * comm_cont,
            len(population)
            )
    smp = sample(population, k=sample_size)

    cluster_set = KMedoids(smp, n_clusters=comm_cont)
    cluster_set = cluster_set.cap(cluster_set.min_len)
    part = Partition()
    edge_set = set()
    for cluster in cluster_set:
        nxt = level + 1
        sub_edge, sub_part = _initialize_communities(graph, param, smp, nxt)
        part.add(sub_part)
        edge_set.update(sub_edge)
    return edge_set, part


def _initialize_leaf_communities(
        population: tp.Iterable[Vector],
        graph: Graph,
        param: Parameters
        ) -> tp.Tuple[tp.Set[tp.Tuple[Vertex, Vertex]], Partition]:
    '''
    Internal function for community initialization.

    This is the counter part for the `_initialize_communities` function.
    This implementation, for leaf communities, generates the edges within.
    '''
    partition = Partition(Vertex(p) for p in population)
    edge_set: tp.Set[tp.Tuple[Vertex, Vertex]] = set()
    for vertex in partition:
        vertex_pool = partition - {vertex} - {e[1] for e in edge_set}
        max_edge_count = min(len(vertex_pool), param.max_within_edge[-1])
        edge_count = max_edge_count
        if (max_edge_count > 1):
            edge_count = rand_uni_range(1, max_edge_count)
        for other_vertex in sample(vertex_pool, k=edge_count):
            edge_set.add((vertex, other_vertex))  # type: ignore
    return edge_set, partition


def initialize_communities(param: Parameters, graph: Graph) -> Graph:
    '''
    This function takes an graph with initialized an initialized vertex set and
    initialize withing it the communities.

    Internally this function will use a set of recursive functions to populate
    the initial communities.
    '''
    edge_set, partition = _initialize_communities(graph, param)
    return Graph(graph.vertex_set, frozenset(edge_set), partition)
