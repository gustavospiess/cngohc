'''Main module.'''


import click
from os import path
from random import shuffle
from collections import Counter


import networkx as nx
import matplotlib.pyplot as plt


from .algo.generator import Parameters, generator
from .algo import rand
from .models import Graph, Vertex


from .validations import relative_inertia, shens_modularity
from .validations import connectivity as _connectivity, diameter as _diameter


def long_docstring(docs: str) -> str:
    return ''.join(line.strip() for line in docs.splitlines())


@click.group()
@click.option(
        '--posfix', '-p', default='', type=str, help='''output file posfix''')
@click.option(
        '--direcory',
        '-d',
        default='out',
        type=str,
        help='''output file direcory''')
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
def connectivity(ctx):
    _connectivity(ctx.obj['graph'])


@check.command()
@click.pass_context
def inertia(ctx):
    relative_inertia(ctx.obj['graph'])


@check.command()
@click.pass_context
def modularity(ctx):
    shens_modularity(ctx.obj['graph'])


@check.command()
@click.pass_context
def diameter(ctx):
    _diameter(ctx.obj['graph'])


@check.command()
@click.pass_context
def edge_distribution(ctx):
    graph = ctx.obj['graph']
    degree = graph.degree_of
    c = Counter(degree[v] for v in graph.vertex_set)
    x = [k for k in sorted(c)]
    y = [c[y] for y in x]
    plt.bar(x, y)
    plt.grid()
    plt.show()


@check.command()
@click.pass_context
def view(ctx):
    graph = ctx.obj['graph']

    nx_graph = nx.Graph()
    nx_graph.add_edges_from(graph.edge_set)

    position = {n: tuple(d*10 for d in n) for n in graph.vertex_set}

    colors = [
            '#7FFFFF', '#00FFFF', '#FF7FFF', '#7F7FFF', '#007FFF', '#FF00FF',
            '#7F00FF', '#0000FF', '#FFFF7F', '#7FFF7F', '#00FF7F', '#FF7F7F',
            '#7F7F7F', '#007F7F', '#FF007F', '#7F007F', '#00007F', '#FFFF00',
            '#7FFF00', '#00FF00', '#FF7F00', '#7F7F00', '#007F00', '#FF0000',
            '#7F0000',
            ]
    shuffle(colors)

    part_color = {p: c for p, c in zip(graph.partition.flat, colors)}

    shared_nodes = [
            v
            for v in graph.vertex_set
            if len(graph.leaf_partitions_of[v]) > 1]
    for p in sorted(graph.partition.flat, key=lambda p: p.level * -1):
        nodes = list(n for n in p if isinstance(n, Vertex))
        if len(nodes) == 0:
            continue
        nx.draw_networkx_nodes(
                nx_graph, position, nodes, node_color=part_color[p])
        plt.plot([0, 0], [-40, 40], lw=3, color='black')
        plt.plot([-40, 40], [0, 0], lw=3, color='black')
        plt.show()
    for p in sorted(graph.partition.flat, key=lambda p: p.level * -1):
        nodes = list(
                n
                for n in p
                if isinstance(n, Vertex) and n not in shared_nodes)
        nx.draw_networkx_nodes(
                nx_graph, position, nodes, node_color=part_color[p])
    nx.draw_networkx_nodes(
            nx_graph, position, shared_nodes, node_color='#000000')
    plt.plot([0, 0], [-40, 40], lw=3, color='black')
    plt.plot([-40, 40], [0, 0], lw=3, color='black')
    plt.show()


@main.command()
@click.option(
        '--seed', '-s', default='17692', type=str, help='''random seed''')
@click.option(
        '--vertex_count', '--N',
        default=Parameters._field_defaults['vertex_count'],
        type=Parameters._field_types['vertex_count'],
        help='''Number of vertexes of the graph''')
@click.option(
        '--min_edge_count', '--MTE',
        default=Parameters._field_defaults['min_edge_count'],
        type=Parameters._field_types['min_edge_count'],
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
