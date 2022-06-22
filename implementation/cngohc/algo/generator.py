'''
This module wraps all the methods for the graph generation.
'''


from ..models import Vertex, Vector, Graph, Partition
from ..models import PartitionBuilder, Edge, edge_set
from .clustering import KMedoids
from .rand import rand_norm, rand_in_range, sample, shuffle, rand_uni
from .rand import rand_pl, rand_edge_within, rand_edge_between

from itertools import chain, repeat, combinations, count
from multiprocessing import Pool
from math import sqrt, ceil
from time import time

import typing as tp


class Parameters(tp.NamedTuple):
    vertex_count: int = 1000  # N
    '''Number of vertexes of the graph'''
    min_edge_count: int = 30000  # MTE
    '''Minimum number of edges of the graph'''
    deviation_sequence: tp.Tuple[float, ...] = (1, 1,)  # A
    '''Sequence of deviation values to initialize the vertexes '''
    homogeneity_indicator: float = 0.1  # theta
    '''Ratio of vertexes to be added by homogeneity'''
    representative_count: int = 20  # NbRep
    '''Number of representatives of a partition'''
    community_count: tp.Tuple[int, ...] = (6, 2,)  # K
    '''
    Sequence of hierarchical communities quantities, the first value indicates
    how many communities will be created at the root of the graph, the second
    indicates how many will be created  inside each of the first ones, and so
    successively.

    The level_count, quantity of levels in the Graph, will be the length of it,
    and the amount of leaf communities will be the product of all those values.
    '''
    max_within_edge: int = 45  # E_max_wth
    '''
    TODO
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
        population: tp.Collection[Vector] = frozenset(),
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
    # cluster_set = cluster_set.cap(cluster_set.min_len)
    part = set()
    edge_set = set()
    for cluster in cluster_set:
        nxt = level + 1
        sub_edge, sub_part = _initialize_communities(
                graph, param, cluster, nxt, id_count)
        part.add(sub_part)
        edge_set.update(sub_edge)
    smp = tuple(rand_uni(p.depht) for p in part)
    edge_set.update({(smp[i-1], smp[i]) for i in range(1, len(smp))})

    return edge_set, Partition(
            part,
            identifier=next(id_count),
            level=level,
            representative_set=frozenset())


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
        vertex_pool = (
                partition -
                {vertex} -
                {e[1] for e in edge_set if vertex in e} -
                {e[0] for e in edge_set if vertex in e})
        max_edge_count = min(len(vertex_pool), param.max_within_edge)
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
    edge_set, partition = _initialize_communities(
            graph, param, graph.vertex_set)
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


def chose_partitions(  # TODO rename to choose
        param: Parameters,
        graph: Graph,
        vertex: Vertex) -> tp.FrozenSet[Partition]:
    '''
    Partition selection fot the batch insertion proccess.

    Givem the parameters, the partition returned may be randomly chosen, or may
    be the ones minimizing the weighed distance.
    '''

    homo = param.homogeneity_indicator
    sorted_pool = sorted((
            (p, min(
                    p.weighed_distance(vertex, rep, homo)
                    for rep in p.representative_set))
            for p in graph.partition.flat
            if len(p.representative_set) > 0
            ), key=lambda pt: pt[1])

    pool = tuple(map(
            lambda pt: pt[0],
            sorted_pool
            ))

    main = rand_pl(pool)
    # main = pool[0]
    max_level = len(param.community_count)
    if main.level == max_level:
        return frozenset({main})

    community_a, community_b = main, main
    while community_a.level != max_level:
        possible_a = tuple(c for c in pool if c in community_a)
        community_a = rand_pl(possible_a)
    community_b = community_a
    community_a = main
    while community_a.level != max_level:
        possible_a = tuple(
                c
                for c in pool
                if c in community_a and c != community_b)
        community_a = rand_pl(possible_a)
    assert community_a != community_b
    return frozenset({community_a, community_b})


def edge_insertion_within(
        param: Parameters,
        graph: Graph,
        vertex: Vertex,
        partition: Partition,
        over_lap: bool
        ) -> tp.FrozenSet[tp.Tuple[Vertex, Vertex]]:
    '''
    Edge generator for the introducing a new member into a community.

    The number of edges is a randomly generated according to a power law.
    The edge is betwen the given vertex and a random member of the community
    choosen randomly with the ´rand_edge_within´.
    '''
    vertex_pool = set(partition.depht)
    max_count = min(len(vertex_pool), param.max_within_edge)+1
    if over_lap:
        max_count = ceil(sqrt(max_count))
    min_count = 1
    edges_within = rand_pl(tuple(i for i in range(min_count+1, max_count+1)))-1
    degree = graph.degree_of.__getitem__

    neighbor_set: tp.Set[Vertex] = set()
    while len(neighbor_set) < edges_within:
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


def find_triples(
        edge_set: tp.AbstractSet[Edge],
        already_checked: tp.Collection[Edge]) -> tp.Iterator[Edge]:
    for edge_a, edge_b in combinations(edge_set, 2):
        if edge_a[0] in edge_b and edge_a[1] in edge_b:
            continue  # they're the same
        if edge_a[0] not in edge_b and edge_a[1] not in edge_b:
            continue  # no shared vertex
        vertex_a = edge_a[0] if edge_a[1] in edge_b else edge_a[1]
        vertex_b = edge_b[0] if edge_b[1] in edge_a else edge_b[1]
        edge_c = (vertex_a, vertex_b)
        edge_c_i = (vertex_b, vertex_a)
        if edge_c not in already_checked and edge_c_i not in already_checked:
            yield min(edge_c, edge_c_i)


def super_choose(generators: tp.List[tp.Iterator[Edge]]) -> tp.Iterator[Edge]:
    while len(generators) > 0:
        for idx in range(len(generators)):
            try:
                yield next(generators[idx])
            except StopIteration:
                del generators[idx]
                shuffle(generators)
                break


def final_edge_generator(graph, level):
    by_level = graph.partition.by_level

    gen = super_choose([
            find_triples(graph.edges_of_part[part], graph.edge_set)
            for part in by_level[level]
            ])
    for tri in gen:
        yield tri


def final_edge_insertino(graph, qtd, level):
    new_edges = set()
    edge_generator = final_edge_generator(graph, level)
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
        ed = edge_insertion_within(
                __parameters,
                __rolling_graph,
                vertex,
                part,
                len(partition_set) > 1)
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
'''
Module wide variable to control the graph for parallelism

It should be read only in most of the code, being written into only with the
generator function.

Being a global variable not being overridden (and the class being immutable) in
a Linux environment, the data do not need to be replicated, being shared
between the jobs. In other environments, it may be replicated.
'''
__parameters: tp.Optional[Parameters] = None
'''
Module wide variable to control the parameters for parallelism

It should be read only in most of the code, being written into only with the
generator function.

Being a global variable not being overridden (and the class being immutable) in
a Linux environment, the data do not need to be replicated, being shared
between the jobs. In other environments, it may be replicated.
'''


def generator(param: Parameters):
    initial_time = time()

    global __parameters
    __parameters = param
    global __rolling_graph
    __rolling_graph = initialize_communities(param, initialize_graph(param))

    new_edges = set()

    count = 0
    for batch in batch_generator(__rolling_graph):
        print(
                f'{round(time()-initial_time, 3)};' +
                f' {count}/{param.vertex_count};' +
                f' {100*count/param.vertex_count}%')
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
                    edge_set(__rolling_graph.edge_set | new_edges),
                    part_builder.build()
                    )
            count += len(batch)

    print(
            f'{round(time()-initial_time, 3)};' +
            f' {count}/{param.vertex_count};' +
            f' {100*count/param.vertex_count}%')

    fei_level = len(__parameters.community_count)
    fei_control = False
    while len(__rolling_graph.edge_set) < param.min_edge_count:
        print(
                f'{round(time()-initial_time, 3)};' +
                f' {len(__rolling_graph.edge_set)}/{param.min_edge_count};' +
                f' {len(__rolling_graph.edge_set)/param.min_edge_count*100}' +
                f'%; {fei_level}')
        final_edges = final_edge_insertino(
                __rolling_graph,
                param.min_edge_count - len(__rolling_graph.edge_set),
                fei_level)

        def max_edges(c):
            return len(c.depht)*(len(c.depht)-1)/2
        fei_control = all(
                len(__rolling_graph.edges_of_part[c]) == max_edges(c)
                for c in __rolling_graph.partition.flat
                if c.level == fei_level)
        if fei_control:
            fei_level -= 1

        __rolling_graph = Graph(
                __rolling_graph.vertex_set,
                edge_set(__rolling_graph.edge_set | final_edges),
                __rolling_graph.partition
                )
    print(
            f'{round(time()-initial_time, 3)}; ' +
            f'{len(__rolling_graph.edge_set)}/{param.min_edge_count}; ' +
            f'{len(__rolling_graph.edge_set)/param.min_edge_count*100}%')


    return __rolling_graph
