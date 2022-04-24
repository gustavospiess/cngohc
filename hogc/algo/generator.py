'''
This module wraps all the methods for the graph generation.
'''


from ..models import Vertex, Vector, Graph, Partition
from .clustering import KMedoids
from .rand import rand_norm, rand_in_range, sample, shuffle
from .rand import rand_threshold, rand_pl, rand_edge_within, rand_edge_between

from itertools import chain, repeat

import typing as tp


class Parameters(tp.NamedTuple):
    vertex_count: int = 1000  # N
    '''Number of vertexes of the graph'''
    min_edge_count: int = 300  # MTE
    '''Minimum number of edges of the graph'''
    deviation_sequence: tp.Tuple[float, ...] = (1, 1,)  # A
    '''Sequence of deviation values to initialize the vertexes '''
    homogeneity_indicator: float = 0.1  # theta
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
    max_within_edge: tp.Tuple[int, ...] = (3, 7, 12)  # E_max_wth
    '''
    Sequence of the max initial edges a vertex will receive when being added to
    a community, the first value is the quantity of edges to be added inside
    the first level community the vertex will be in, the second value for the
    second level community and so on.
    This should be a sequence of length equal to the level count of the graph
    plus one, as for initialization purposes, the whole graph is considered r
    community.
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
        return _initialize_leaf_communities(population, graph, param, level)
    comm_cont = param.community_count[level]

    sample_size = min(
            (param.representative_count ** (max_level-level)) * comm_cont,
            len(population)
            )
    smp = sample(population, k=sample_size)

    cluster_set = KMedoids(smp, n_clusters=comm_cont)
    cluster_set = cluster_set.cap(cluster_set.min_len)
    part = set()
    edge_set = set()
    for cluster in cluster_set:
        nxt = level + 1
        sub_edge, sub_part = _initialize_communities(graph, param, smp, nxt)
        part.add(sub_part)
        edge_set.update(sub_edge)
    return edge_set, Partition(
            part,
            level=level,
            representative_set=frozenset(Vertex(s) for s in chain(*edge_set)))


def _initialize_leaf_communities(
        population: tp.Iterable[Vector],
        graph: Graph,
        param: Parameters,
        level: int
        ) -> tp.Tuple[tp.Set[tp.Tuple[Vertex, Vertex]], Partition]:
    '''
    Internal function for community initialization.

    This is the counter part for the `_initialize_communities` function.
    '''
    members = frozenset(Vertex(p) for p in population)
    partition = Partition(members, level=level, representative_set=members)
    edge_set: tp.Set[tp.Tuple[Vertex, Vertex]] = set()
    for vertex in partition:
        vertex_pool = partition - {vertex} - {e[1] for e in edge_set}
        max_edge_count = min(len(vertex_pool), param.max_within_edge[-1])
        edge_count = max_edge_count
        if (max_edge_count > 1):
            edge_count = rand_in_range(range(1, max_edge_count))
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


def batch_generator(graph: Graph) -> tp.Generator[tp.Set[Vertex], None, None]:
    '''
    Generate successive random sized sets of vertexes from the givem graph.

    The union of all the sets generated is equal to the original vertex set of
    the graph.
    '''
    to_add = set(graph.zero_degree_vertex)
    while to_add:
        smp_count = max(1, rand_in_range(range(len(to_add))))
        smp = sample(to_add, k=smp_count)
        to_add -= smp
        yield smp


def chose_partitions(
        param: Parameters,
        graph: Graph,
        vertex: Vertex) -> tp.FrozenSet[Partition]:
    '''
    Partition selection fot the batch insertion proccess.

    Givem the parameters, the partition returned may be randomly chosen, or may
    be the ones minimizing the weighed distance.
    '''
    heterogenic = rand_threshold(param.homogeneity_indicator)

    possible: tp.Tuple[Partition, ...]
    if heterogenic:
        possible = tuple(graph.partition.flat)
    else:
        possible = tuple(sorted(
                graph.partition.flat,
                key=lambda p: min(
                    p.weighed_distance(vertex, rep)
                    for rep in p.representative_set)
                ))

    return frozenset(rand_pl(possible) for _ in range(len(possible)//2))


def edge_insertion_within(
        param: Parameters,
        graph: Graph,
        vertex: Vertex,
        partition: Partition,
        ) -> tp.FrozenSet[tp.Tuple[Vertex, Vertex]]:
    '''
    Edge generator for the introducing a new member into a community.

    The number of edges is a randomly generated according to a power law.
    The edge is betwen the given vertex and a random member of the community
    choosen randomly with the ´rand_edge_within´.
    '''
    level: int = partition.level
    vertex_pool = set(partition.depht)
    max_count = min(len(vertex_pool), param.max_within_edge[level])
    edges_within = rand_pl(tuple(i for i in range(1, max_count)))

    degree = graph.degree_of.__getitem__

    neighbor_set: tp.Set[Vertex] = set()
    for _ in range(edges_within):
        other = rand_edge_within(vertex_pool, degree)
        vertex_pool.remove(other)
        neighbor_set.add(other)
    return frozenset((vertex, n) for n in neighbor_set)


def edge_insertion_between(
        param: Parameters,
        graph: Graph,
        vertex: Vertex,
        ignored: tp.Set[Partition],
        limit: int,
        ) -> tp.FrozenSet[tp.Tuple[Vertex, Vertex]]:
    '''
    TODO: doc
    '''

    partition_pool = graph.partition.flat - ignored
    vertex_pool: tp.Set[tp.Tuple[Vertex, Partition]] = set(chain(*tuple(
            zip(p.representative_set, repeat(p)) for p in partition_pool)))

    max_count = min(limit, param.max_between_edge, len(vertex_pool))
    if max_count == 0:
        return frozenset()

    edges_between = rand_pl(tuple(i for i in range(1, max_count+1)))

    neighbor_set: tp.Set[Vertex] = set()
    while len(neighbor_set) < edges_between:
        other = rand_edge_between(
                vertex_pool,
                lambda pair: pair[1].weighed_distance(vertex, pair[0]))
        vertex_pool.remove(other)
        neighbor_set.add(other[0])
    return frozenset((neighbor, vertex,) for neighbor in neighbor_set)


T = tp.TypeVar('T')
EdgeSet = tp.FrozenSet[tp.Tuple[T, T]]
Triplets = tp.FrozenSet[T]


def find_triples(edge_set: EdgeSet) -> tp.Generator[Triplets, None, None]:
    for edge_a in shuffle(edge_set):
        for edge_b in shuffle(edge_set):
            if edge_a[0] in edge_b and edge_a[1] in edge_b:
                continue  # same edge
            if edge_a[0] not in edge_b and edge_a[1] not in edge_b:
                continue  # no shared vertex
            yield frozenset((*edge_a, *edge_b))
