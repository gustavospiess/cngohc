'''
This module wraps all the methods for the graph generation.
'''


from ..models import Vertex, Vector, Graph, Partition, PartitionBuilder, Edge
from .clustering import KMedoids
from .rand import rand_norm, rand_in_range, sample, shuffle
from .rand import rand_threshold, rand_pl, rand_edge_within, rand_edge_between

from itertools import chain, repeat, combinations, count
from multiprocessing import Pool

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
        level: int = 0,
        id_count: tp.Iterator[int] = count()
        ) -> tp.Tuple[tp.Set[tp.Tuple[Vertex, Vertex]], Partition]:
    '''
    Internal function for community initialization.

    It recursively runs a clustering method and initialize the communities.
    All the vertexes are in leaf communities.

    All the leaf communities are connected components of the graph.

    The return is the set of edges generates, as well as the partitions.
    '''

    if not population:
        return _initialize_communities(
                graph, param, graph.vertex_set, level, id_count)

    max_level = len(param.community_count)
    if level == max_level:
        return _initialize_leaf_communities(
                population, graph, param, level, id_count)
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
        sub_edge, sub_part = _initialize_communities(
                graph, param, smp, nxt, id_count)
        part.add(sub_part)
        edge_set.update(sub_edge)
    return edge_set, Partition(
            part,
            identifier=next(id_count),
            level=level,
            representative_set=frozenset(Vertex(s) for s in chain(*edge_set)))


def _initialize_leaf_communities(
        population: tp.Iterable[Vector],
        graph: Graph,
        param: Parameters,
        level: int,
        id_count: tp.Iterator[int]
        ) -> tp.Tuple[tp.Set[tp.Tuple[Vertex, Vertex]], Partition]:
    '''
    Internal function for community initialization.

    This is the counter part for the `_initialize_communities` function.
    '''
    members = frozenset(Vertex(p) for p in population)
    partition = Partition(
            members,
            identifier=next(id_count),
            level=level,
            representative_set=members)
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
    Given a graph with a number of zero degree vertexes, it yields sets of
    those vertexes, without repeating, and eventually yielding every one.

    The size of this batches are defined as growing vallues, initially with
    1 plus half the number of communities, and doubleling it untill pass 1_000.
    when it passes 5_000, the batch size will be randombly choosen uniformly
    between 5_000 and 10_000.

    The last batch will be the padding, having a size ranging from zero to 999.
    '''
    # to_add = set(graph.zero_degree_vertex)
    to_add = set(graph.vertex_set - graph.partition.depht)
    sample_size = 1 + (len(graph.partition.flat) // 2)
    while len(to_add) > sample_size and sample_size < 5000:
        sampled_data = sample(to_add, k=sample_size)
        to_add -= sampled_data
        yield sampled_data
        sample_size = min(sample_size*2, len(to_add))
    while len(to_add) > 5000:
        sample_size = rand_in_range(range(
            min(5000, len(to_add)),
            min(10000, len(to_add))))
        sampled_data = sample(to_add, k=sample_size)
        to_add -= sampled_data
        yield sampled_data
    yield to_add


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

    possible: tp.Sequence[Partition]
    if heterogenic:
        possible = list(graph.partition.flat)
        shuffle(possible)
        possible = tuple(possible)
    else:
        possible = tuple(sorted(
                graph.partition.flat,
                key=lambda p: min(
                    p.weighed_distance(vertex, rep)
                    for rep in p.representative_set)
                ))

    quantity = rand_pl(tuple(i for i in range(1, len(possible)//2)))
    return frozenset(rand_pl(possible) for _ in range(quantity))


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
    edges_within = rand_pl(tuple(i for i in range(1, max_count+1)))

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
        ignored: tp.AbstractSet[Partition],
        limit: int,
        ) -> tp.FrozenSet[tp.Tuple[Vertex, Vertex]]:
    '''
    Edge generator for linking an vertex into communities it do not belong.

    The number of edges is a randomly generated according to a power law.
    The edge is betwen the given vertex and a random community representative
    choosen with the ´rand_edge_between´.
    '''

    partition_pool = graph.partition.flat - ignored
    vertex_pool = frozenset(chain(*tuple(
            zip(p.representative_set, repeat(p)) for p in partition_pool)))

    max_count = min(limit, param.max_between_edge, len(vertex_pool))
    edges_between = rand_pl(tuple(i for i in range(0, max_count+1)))

    neighbor_set = rand_edge_between(vertex_pool, vertex, edges_between)
    return frozenset((neighbor, vertex,) for neighbor in neighbor_set)


def find_triples(edge_set: tp.AbstractSet[Edge]) -> tp.Iterator[Edge]:
    for edge_a, edge_b in combinations(edge_set, 2):
        if edge_a[0] in edge_b and edge_a[1] in edge_b:
            continue  # they're the same
        if edge_a[0] not in edge_b and edge_a[1] not in edge_b:
            continue  # no shared vertex
        edge_c = (
                edge_a[0] if edge_a[1] in edge_b else edge_a[1],
                edge_b[0] if edge_b[1] in edge_a else edge_b[1],
                )
        edge_c_inversed = tuple(reversed(edge_c))
        if edge_c not in edge_set and edge_c_inversed not in edge_set:
            yield edge_c


def super_choose(generators: tp.List[tp.Iterator[Edge]]) -> tp.Iterator[Edge]:
    while len(generators) > 0:
        for idx in range(len(generators)):
            try:
                yield next(generators[idx])
            except StopIteration:
                del generators[idx]
                shuffle(generators)
                break


def final_edge_generator(graph):
    by_level = graph.partition.by_level
    triplet_gen = iter(tuple())
    for level in reversed(sorted(by_level)):
        triplet_gen = chain(triplet_gen, super_choose([
            find_triples(graph.edges_of_part[part])
            for part in by_level[level]
            ]))
    for tri in triplet_gen:
        if tri not in graph.edge_set:
            yield tri


def final_edge_insertino(graph, qtd):
    new_edges = set()
    edge_generator = final_edge_generator(graph)
    while len(new_edges) < qtd:
        pending_qtd = qtd - len(new_edges)

        t = [e for e, i in zip(edge_generator, range(pending_qtd))]
        new_edges.update(t)
        if len(t) < pending_qtd:
            return new_edges
    return new_edges


def introduce_vertex(vertex: Vertex):
    if __parameters is None or __rolling_graph is None:
        raise ValueError('invalid state call')
    partition_set = chose_partitions(__parameters, __rolling_graph, vertex)
    vertex_neighboors: tp.Set[Edge] = set()
    for part in partition_set:
        ed = edge_insertion_within(__parameters, __rolling_graph, vertex, part)
        vertex_neighboors.update(ed)
    ed = edge_insertion_between(
            __parameters,
            __rolling_graph,
            vertex,
            partition_set,
            len(vertex_neighboors))
    vertex_neighboors.update(ed)
    return (frozenset(vertex_neighboors),
            tuple(p.identifier for p in partition_set),
            vertex)


__rolling_graph: tp.Optional[Graph] = None
'''TODO: Doc this'''
__parameters: tp.Optional[Parameters] = None
'''TODO: Doc this'''


def generator(param: Parameters):
    tot = 0

    global __parameters
    __parameters = param
    global __rolling_graph
    __rolling_graph = initialize_communities(param, initialize_graph(param))

    new_edges = set()
    for batch in batch_generator(__rolling_graph):
        with Pool() as p:
            proccess_batch = p.imap_unordered(introduce_vertex, batch)
            part_builder = PartitionBuilder(
                    __rolling_graph.partition, param.representative_count)
            for vertex_neighboors, partition_ids, vertex in proccess_batch:
                partition_set = {
                        partition
                        for partition in __rolling_graph.partition.flat
                        if partition.identifier in partition_ids}
                new_edges.update(vertex_neighboors)
                part_builder.add_all(partition_set, vertex)
            __rolling_graph = Graph(
                    __rolling_graph.vertex_set,
                    frozenset(__rolling_graph.edge_set | new_edges),
                    part_builder.build()
                    )
            tot += len(batch)

    while len(__rolling_graph.edge_set) < param.min_edge_count:
        final_edges = final_edge_insertino(
                __rolling_graph,
                param.min_edge_count - len(__rolling_graph.edge_set))
        __rolling_graph = Graph(
                __rolling_graph.vertex_set,
                frozenset(__rolling_graph.edge_set | final_edges),
                __rolling_graph.partition
                )

    return __rolling_graph
