'''Main module.'''


import click
from os import path


import networkx as nx
import matplotlib.pyplot as plt


from .algo.generator import Parameters, generator
from .algo import rand
from .models import Graph


from .validations import connectivity, relative_inertia


def long_docstring(docs: str) -> str:
    return ''.join(line.strip() for line in docs.splitlines())


@click.group()
@click.option('--posfix', '-p', default='', type=str, help='''output file posfix''')
@click.option('--direcory', '-d', default='out', type=str, help='''output file direcory''')
@click.pass_context
def main(ctx, posfix, direcory):
    ctx.ensure_object(dict)
    if posfix:
        posfix = '_' + posfix
    ctx.obj['posfix'] = posfix
    ctx.obj['direcory'] = direcory
    pass


@main.group()
@click.pass_context
def check(ctx):
    posfix = ctx.obj['posfix']
    direcory = ctx.obj['direcory']

    graph = Graph()

    with open(path.join(direcory, f'vertex{posfix}.txt'), 'r') as v_buffer:
        graph = graph.read_vertex_from_buffer(v_buffer)
    with open(path.join(direcory, f'edge{posfix}.txt'), 'r') as e_buffer:
        graph = graph.read_edge_from_buffer(e_buffer)
    with open(path.join(direcory, f'partition{posfix}.json'), 'r') as p_buffer:
        graph = graph.read_partition_from_buffer(p_buffer)

    ctx.obj['graph'] = graph


@check.command()
@click.pass_context
def every_community_is_connected(ctx):
    connectivity(ctx.obj['graph'])


@check.command()
@click.pass_context
def relative_inertia_is_lower(ctx):
    relative_inertia(ctx.obj['graph'])


@check.command()
@click.pass_context
def view(ctx):
    graph = ctx.obj['graph']

    nx_graph = nx.Graph()
    nx_graph.add_edges_from(graph.edge_set)
    # nx.draw(nx_graph, pos={n: tuple(d*10 for d in n) for n in graph.vertex_set})
    nx.draw(nx_graph, pos=nx.spring_layout(nx_graph, iterations=1000))
    # nx.draw(nx_graph, pos=nx.nx_pydot.graphviz_layout(nx_graph, prog='neato'))
    plt.show()

    # comunity_graph = nx.Graph()
    # community_set = set(c for c in graph.partition.flat if c != graph.partition)
    # for community in community_set:
    #     for community_b in community_set:
    #         if community_b != community:
    #             new_node = 100000 * max(community.identifier, community_b.identifier) + min(community.identifier, community_b.identifier)
    #             comunity_graph.add_edge(
    #                     community.identifier,
    #                     new_node,
    #                     weight=len(set(community.depht) & set(community_b.depht)))
    #             comunity_graph.add_edge(
    #                     community_b.identifier,
    #                     new_node,
    #                     weight=len(set(community.depht) & set(community_b.depht)))
    # nx.draw(comunity_graph, pos=nx.nx_pydot.graphviz_layout(comunity_graph, prog='neato'))
    # nx.draw(comunity_graph)
    # nx.draw(comunity_graph, pos=nx.bipartite_layout(comunity_graph, graph.partition.flat))
    # nx.draw(comunity_graph, pos=nx.spring_layout(comunity_graph, iterations=1000))
    # plt.show()

    # nx_bi_graph = nx.Graph()
    # for vertex in graph.vertex_set:
    #     for community in graph.partitions_of[vertex]:
    #         nx_bi_graph.add_edge(vertex, community)
    # nx.draw(nx_bi_graph, pos=nx.bipartite_layout(nx_bi_graph, graph.vertex_set))
    # plt.show()


@main.command()
@click.option('--seed', '-s', default='17692', type=str, help='''random seed''')
@click.option(
        '--vertex_count', '--N',
        default=Parameters._field_defaults['vertex_count'],
        type=Parameters._field_types['vertex_count'],
        help='''Number of vertexes of the graph''')
@click.option(
        '--min_edge_count', '--MTE',
        default=Parameters._field_defaults['min_edge_count'], type=Parameters._field_types['min_edge_count'],
        help='''Minimum number of edges of the graph''')
@click.option(
        '--deviation_sequence', '--A',
        default=Parameters._field_defaults['deviation_sequence'],
        # type=Parameters._field_types['deviation_sequence'],
        multiple=True, type=float,
        help='''Sequence of deviation values to initialize the vertexes ''''')
@click.option(
        '--homogeneity_indicator', '--theta',
        default=Parameters._field_defaults['homogeneity_indicator'],
        type=Parameters._field_types['homogeneity_indicator'],
        help='''Ratio of vertexes to be added by homogeneity''')
@click.option(
        '--representative_count', '--NbRep',
        default=Parameters._field_defaults['representative_count'],
        type=Parameters._field_types['representative_count'],
        help='''Number of representatives of a partition''')
@click.option(
        '--community_count', '--K',
        default=Parameters._field_defaults['community_count'],
        # type=Parameters._field_types['community_count'],
        multiple=True, type=int,
        help=long_docstring('''
                Sequence of hierarchical communities quantities, the first
                value indicates how many communities will be created at the
                root of the graph, the second indicates how many will be
                created  inside each of the first ones, and so successively.
                The level_count, quantity of levels in the Graph, will be the
                length of it, and the amount of leaf communities will be the
                product of all those values.'''))
@click.option(
        '--max_within_edge', '--E_max_wth',
        default=Parameters._field_defaults['max_within_edge'],
        # type=Parameters._field_types['max_within_edge'],
        multiple=True, type=int,
        help=long_docstring('''
                Sequence of the max initial edges a vertex will receive when
                being added to a community, the first value is the quantity of
                edges to be added inside the first level community the vertex
                will be in, the second value for the second level community and
                so on.  This should be a sequence of length equal to the level
                count of the graph plus one, as for initialization purposes,
                the whole graph is considered a community.'''))
@click.option(
        '--max_between_edge', '--E_max_btw',
        default=Parameters._field_defaults['max_between_edge'],
        type=Parameters._field_types['max_between_edge'],
        help=long_docstring('''
                Maximum quantity of initial edges a vertex will receive on
                addition to a community linking it to outside the
                community.'''))
@click.pass_context
def generate(
        ctx,
        seed,
        vertex_count,
        min_edge_count,
        deviation_sequence,
        homogeneity_indicator,
        representative_count,
        community_count,
        max_within_edge,
        max_between_edge
        ):
    '''TODO'''

    rand.DEFAULT_RANDOM.seed(seed)

    params = Parameters(
        vertex_count,
        min_edge_count,
        deviation_sequence,
        homogeneity_indicator,
        representative_count,
        community_count,
        max_within_edge,
        max_between_edge)
    graph = generator(params)

    posfix = ctx.obj['posfix']
    direcory = ctx.obj['direcory']

    with open(path.join(direcory, f'vertex{posfix}.txt'), 'w') as v_buffer:
        graph.write_vertex_to_buffer(v_buffer)
    with open(path.join(direcory, f'edge{posfix}.txt'), 'w') as e_buffer:
        graph.write_edge_to_buffer(e_buffer)
    with open(path.join(direcory, f'partition{posfix}.json'), 'w') as p_buffer:
        graph.write_partition_to_buffer(p_buffer)
