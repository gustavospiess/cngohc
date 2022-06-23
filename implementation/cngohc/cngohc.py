'''Main module.'''


import click
from os import path
from random import shuffle
from collections import Counter
from itertools import product, repeat


import networkx as nx
import matplotlib.pyplot as plt


from .algo.generator import Parameters, generator
from .algo import rand
from .models import Graph, Vertex, Partition


from .validations import relative_inertia, shens_modularity, homophily as homo
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
def homophily(ctx):
    homo(ctx.obj['graph'])


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
def data(ctx):
    graph = ctx.obj['graph']
    da = dict()
    # print('data:')
    for p in sorted(graph.partition.flat, key=lambda p: p.level):
        # print(
        #         # f'partition id {p.identifier}; ' +
        #         # f'level {p.level}; ' +
        #         # f'len {len(p.depht)}; ' +
        #         f'wth degree {len(graph.edges_of_part[p])*2}; ' +
        #         f'degree {sum(graph.degree_of[v] for v in p.depht)};')
        d = len(p.depht)
        d_ = sum(len(s.depht) for s in p if isinstance(s, Partition))
        da[p] = (d, d_)
    max_l = max(p.level for p in graph.partition.flat)
    for l in range(0, max_l):
        a = sum(da[p][0] for p in da if p.level == l)
        b = sum(da[p][1] for p in da if p.level == l)
        print(l, b-a)


@check.command()
@click.pass_context
def view(ctx):
    graph = ctx.obj['graph']

    nx_graph = nx.Graph()
    nx_graph.add_edges_from(graph.edge_set)


    colors = [
            'red', 'lime', 'green', 'yellow', 'blue', 'purple', 'orange', 'pink', 'cyan',
            'teal', 'fuchsia', 'brown',
            'gray', 'olive', 'crimson', 'tan', 'navy', 'chocolate']
    color = (c for _ in range(100) for  c in colors) 
    shape = 's^>v<dph8o'
    part_color = dict()
    part_color[graph.partition] = ('o', 'white')
    for commuinty_a, _shape in zip(graph.partition, shape):
        part_color[commuinty_a] = (_shape, 'white')
        if isinstance(commuinty_a, Partition):
            for commuinty_b in commuinty_a:
                _color = next(color)
                part_color[commuinty_b] = (_shape, _color)

    style_map = dict()
    for p in sorted(graph.partition.flat, key=lambda p: p.level):
        if p.level > 2:
            break
        for v in p.depht:
            if all(sub in p for sub in graph.leaf_partitions_of[v]) or (len(graph.leaf_partitions_of[v]) == 1):
                style_map[v] = part_color[p]

    position = {n: tuple(d*10 for d in n) for n in graph.vertex_set}
    for i in (0, 5):
        if i > 0:
            position = nx.spring_layout(nx_graph, k=2, pos=position, iterations=i)

        for v in graph.partition.depht:
            nx.draw_networkx_nodes(
                    nx_graph,
                    position,
                    (v,),
                    node_size=150,
                    node_color=style_map[v][1],
                    node_shape=style_map[v][0],
                    edgecolors='#000000')
        nx.draw_networkx_edges(nx_graph, pos=position, width=0.2)
        plt.show()
        nx.draw_networkx_edges(nx_graph, pos=position, width=0.1)
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
        type=Parameters._field_types['max_within_edge'],
        # multiple=True, type=int,
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
