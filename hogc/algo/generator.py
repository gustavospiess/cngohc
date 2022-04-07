'''
This module wraps all the methods for the graph generation.
'''


from ..models import Vertex, Graph
from .rand import rand_norm

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
    '''Number oif representatieves of a partition'''
    comunity_count: tp.Tuple[int, ...] = (3, 2,)  # K
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
    Sequence of the max initial edges a vertex will recieve when being added to
    a community, the first value is the quantity of edges to be added inside
    the first level community the vertex will be in, the second value for the
    second level community and so on.
    This should be a sequence of length equal to the level count of the graph.
    '''
    max_between_edge: int = 3  # E_max_wth
    '''Maximum quantity of initial edges a vertex will recieve on addition to a
    community linking it to outside the comunity.'''


def make_vertex(deviation_seq: tp.Sequence[float]) -> Vertex:
    return Vertex(rand_norm(0, dev) for dev in deviation_seq)


def initialize_graph(p: Parameters) -> Graph:
    g = Graph()
    g.vertex_set.update(
            make_vertex(p.deviation_sequence) for _ in range(p.vertex_count))
    return g
