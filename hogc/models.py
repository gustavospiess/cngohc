'''This module defines the base models to the process'''


from json import dump as json_dump, load as json_load
from csv import reader as csv_reader, writer as csv_writer

from itertools import chain, count
from functools import lru_cache, wraps
from collections import Counter
from math import sqrt as square_root

import typing as tp
_T = tp.TypeVar('_T')


# TODO: move otherwhere
def _unchain(
        data: tp.Iterable[_T]) -> tp.Generator[tp.Tuple[_T, _T], None, None]:
    '''
    Generator to unchain an iterable constructed of pairs.

    It is to be the inverse of `functools.chain(*data)` where `data` is a list
    of tuples.
    '''
    it = iter(data)
    while True:
        try:
            yield (next(it), next(it))
        except (StopIteration):
            return


def cached_generator(gen):
    @lru_cache
    @wraps(gen)
    def set_wraped(*args, **kwargs):
        return frozenset(gen(*args, **kwargs))
    return set_wraped


class Vector(tp.Tuple[float, ...]):
    '''
    This class implements some utilities, so you can add them and subtract
    vectors.

    You can also get their absolute value (the linear distance from it to the
    origin), optimizing the computing for the absolute.

    When using `a > b` it
    will return the same as `abs(a) > abs(b)` but it will not calculate the
    square roots.

    It has a class method for averaging an iterable  of vectors.
    '''

    def __abs__(self) -> float:
        return square_root(sum(x**2 for x in self))

    def __sub__(self, other) -> 'Vector':
        return Vector(s-o for s, o in zip(self, other))

    def __add__(self, other) -> 'Vector':
        return Vector(s+o for s, o in zip(self, other))

    def __ge__(self, other) -> bool:
        if self == other:
            return True
        return sum(s**2 for s in self) > sum(o**2 for o in other)

    def __lt__(self, other) -> bool:
        return not self >= other

    def __le__(self, other) -> bool:
        return self == other or self < other

    @classmethod
    def avg(cls, data: tp.Sequence['Vector']) -> 'Vector':
        qt = len(data)
        return cls(sum(d)/qt for d in zip(*data))


class Vertex(Vector):
    '''
    Representation of attributed vertex for the graphs
    '''

    def to_raw(self) -> tp.Tuple[float, ...]:
        return self

    @classmethod
    def from_raw(
            cls, data: tp.Iterable[tp.Union[tp.Text, int, float]]) -> 'Vertex':
        return cls(float(f) for f in data)


PartitionMember = tp.Union[Vertex, 'Partition']
'''
Type union for what can compose a Partition.
Used for type checking only.
'''


class Partition(tp.FrozenSet[PartitionMember]):
    '''
    Representation of a graph partition.

    It implements a static hash function, but it is mutable.
    It needs to be hash enabled to be used as a recursive type, as the
    partitions of a graph may contain other partitions.

    It also override the __contains__ method from `set`, so it checks if the
    given value is in itself or in some descendant.
    '''
    __slots__ = [
            '__identifier',
            '__level',
            '__weigh_vector',
            '__representative_set']

    session_id = count()

    def __new__(
            cls, members: tp.Iterable[PartitionMember] = tuple(),
            *args, **kwargs):
        return super().__new__(cls, members)  # type: ignore

    def __init__(
            self,
            members: tp.Iterable[PartitionMember] = tuple(),
            identifier: tp.Optional[int] = None,
            level: int = 0,
            *,
            weigh_vector: Vector = Vector(),
            representative_set: tp.FrozenSet[Vertex] = frozenset(),):
        super
        if identifier is None:
            identifier = next(Partition.session_id)
        self.__identifier: int = identifier
        '''
        This is used to force a static hash, if not informed in initialization
        `id(self)` will be used
        '''
        self.__level = level
        '''
        Internal definition of level, zero meaning the intance is a root.
        '''
        self.__representative_set = representative_set
        self.__weigh_vector = weigh_vector or self.inverse_max_inertia_axis()
        '''
        Vector of weighs to consider while calculating the distance /
        compatibility of a pair of Vertexes for the context of this partition.
        '''

    @property
    def identifier(self) -> int:
        return self.__identifier

    @property
    def level(self) -> int:
        return self.__level

    def __hash__(self) -> int:
        return super().__hash__() + 31 * hash(self.identifier)

    def __contains__(self, vl: object) -> bool:
        for member in self:
            if member == vl:
                return True
            if isinstance(member, Partition) and vl in member:
                return True
        return False

    @property
    def depht(self) -> tp.FrozenSet[Vertex]:
        return self.__depht()

    @cached_generator
    def __depht(self) -> tp.Generator[Vertex, None, None]:
        '''
        Generator method for traversing the Partition and its sub partitions
        and yield the Vertex.
        '''
        for member in self:
            if isinstance(member, Partition):
                yield from member.depht
            else:
                yield member

    @property
    def flat(self) -> tp.FrozenSet['Partition']:
        return self.__flat()

    @cached_generator
    def __flat(self) -> tp.Generator['Partition', None, None]:
        '''
        Recursive generator for the partitions contained in the hierarchy.
        '''
        yield self
        for member in self:
            if isinstance(member, Partition):
                yield from member.flat

    def to_raw(self) -> tp.Dict[str, tp.Any]:
        '''
        Extract the data into a `Raw` and JSON serializable data structure for
        saving the data.

        It is optimized for readability and ease of use, not for performance.
        '''
        return {'identifier': self.identifier,
                'members': tuple(m.to_raw() for m in self),
                'weigh_vector': self.weigh_vector,
                'representative_set': tuple(self.__representative_set),
                'level': self.level
                }

    @classmethod
    def from_raw(cls, raw: tp.Dict[str, tp.Any]) -> 'Partition':
        '''
        Class method for generating a new instance of `Partition` from the raw
        data extracted from `to_raw`.
        '''
        if not isinstance(raw, tp.Mapping):
            raise TypeError('Invalid data structure')

        members: tp.Set[tp.Union[Vertex, 'Partition']] = set()
        if 'identifier' not in raw or 'members' not in raw:
            raise TypeError('Invalid data structure')
        for m in raw['members']:
            if isinstance(m, dict):
                members.add(cls.from_raw(m))
            elif isinstance(m, tp.Iterable):
                members.add(Vertex(m))
            else:
                raise TypeError('Invalid data structure')

        raw['members'] = frozenset(members)
        raw['representative_set'] = frozenset(
                map(Vertex, raw['representative_set']))

        if 'weigh_vector' not in raw:
            raise TypeError('Invalid data structure')
        if 'level' not in raw:
            raise TypeError('Invalid data structure')
        raw['weigh_vector'] = Vector(raw['weigh_vector'])

        return cls(**raw)

    @property
    def weigh_vector(self) -> Vector:
        return self.__weigh_vector

    @property
    def representative_set(self) -> tp.FrozenSet[Vertex]:
        return self.__representative_set

    def weighed_distance(self, representative: Vertex, other: Vertex) -> float:
        '''
        Calculates and returns a compatibility measure.
        This measure could be an float getting the square of the euclidian
        distance, or other implementation.
        '''
        return sum(
                (rep-oth)**2*wei
                for rep, oth, wei
                in zip(representative, other, self.weigh_vector)
                )

    @lru_cache
    def inverse_max_inertia_axis(self) -> Vector:
        '''
        Calculates and returns a Vector indicating the least relevant axis in
        the inertia of the partition.

        It calculates the inertia of the cluster as a total and considering
        each axis by itself. Then it constructs a vector with the difference
        between the axis inertia and the total inertia, and normalizes it so
        the sum of its part is one.
        '''
        depht = tuple(self.depht)
        center = Vector.avg(depht)
        diffs = tuple(d - center for d in depht)
        if len(center) == 1 or len(diffs) <= 1:
            return Vector((1,))
        local_inertia = (tuple(d**2 for d in v) for v in diffs)
        inertia = tuple(sum(axis) for axis in zip(*local_inertia))
        normal_inertia = (axis/sum(inertia) for axis in inertia)
        inverted = tuple(1 - axis for axis in normal_inertia)
        return Vector(i/sum(inverted) for i in inverted)


class Graph(tp.NamedTuple):
    '''
    Representation of a mutable graph.

    ------

    It's composed of the set `v` for the vertexes, the set `e` for the edges,
    and the set `p` for the partitions.
    '''

    vertex_set: tp.FrozenSet[Vertex] = frozenset()
    edge_set: tp.FrozenSet[tp.Tuple[Vertex, Vertex]] = frozenset()
    partition: Partition = Partition()

    @property
    def neighbors_of(self) -> '_NeighborsOf':
        '''
        Mapping property of the graph that returns a generator of vertexes
        next to the index.
        '''
        return _NeighborsOf(self)

    @property
    def partitions_of(self) -> '_PartitionsOf':
        '''
        Mapping property of the graph that returns a generator of partitions,
        yeilding every partition the vertex is inside.
        '''
        return _PartitionsOf(self)

    @property
    def degree_of(self) -> '_DegreeOf':
        return _DegreeOf(self)

    @property
    def zero_degree_vertex(self) -> tp.Tuple[Vertex, ...]:
        return tuple(v for v in self.vertex_set if self.degree_of[v] == 0)

    def write_partition_to_buffer(self, buf: tp.IO[str]):
        json_dump(self.partition.to_raw(), buf)

    def write_vertex_to_buffer(self, buf: tp.IO[str]):
        csv_writer(buf).writerows(self.vertex_set)

    def write_edge_to_buffer(self, buf: tp.IO[str]):
        csv_writer(buf).writerows(chain(*self.edge_set))

    def read_partition_from_buffer(self, buf: tp.IO[str]):
        return Graph(
                vertex_set=self.vertex_set,
                edge_set=self.edge_set,
                partition=Partition.from_raw(json_load(buf)),
                )

    def read_vertex_from_buffer(self, buf: tp.Iterable[tp.Text]):
        reader = csv_reader(buf)
        return Graph(
                vertex_set=frozenset(Vertex.from_raw(l) for l in reader),
                edge_set=self.edge_set,
                partition=self.partition
                )

    def read_edge_from_buffer(self, buf: tp.Iterable[tp.Text]):
        unchain = _unchain(csv_reader(buf))
        edge_set = frozenset(
                (Vertex.from_raw(p), Vertex.from_raw(q)) for p, q in unchain)
        res = Graph(
                vertex_set=self.vertex_set,
                edge_set=edge_set,
                partition=self.partition)
        return res


class _GraphMapping(
        tp.Mapping[Vertex, _T],
        tp.Generic[_T]):
    '''
    Abstract map of properties in a graph.
    '''
    __slots__ = ['__graph']

    def __init__(self, graph: Graph):
        self.__graph = graph

    @property
    def graph(self) -> Graph:
        return self.__graph

    def __iter__(self) -> tp.Iterator[Vertex]:
        return iter(self.graph.vertex_set)

    def __len__(self) -> int:
        return len(self.graph.vertex_set)

    def __hash__(self) -> int:
        return 23 + 97 * id(self)

    @cached_generator
    def __recursive_gen(
            self,
            item: Vertex,
            stack: tp.Tuple[Partition, ...] = tuple()):
        for sub in stack[-1]:
            if isinstance(sub, Partition):
                part = self.__recursive_gen(item, stack + (sub,))
                if part:
                    return part
            elif item == sub:
                return stack
        return tuple()


class _PartitionsOf(_GraphMapping[tp.Iterable[Partition]]):
    '''Graph mapping for vertex communities'''
    def __getitem__(self, item: Vertex) -> tp.Iterable[Partition]:
        it = self.__recursive_gen(item, (self.graph.partition,))
        return it

    @cached_generator
    def __recursive_gen(
            self,
            item: Vertex,
            stack: tp.Tuple[Partition, ...] = tuple()):
        for sub in stack[-1]:
            if isinstance(sub, Partition):
                yield from self.__recursive_gen(item, stack + (sub,))
            elif item == sub:
                yield from stack


class _NeighborsOf(_GraphMapping[tp.Iterable[Vertex]]):
    '''Graph mapping for vertex neibors'''
    @cached_generator
    def __getitem__(self, item: Vertex) -> tp.Iterable[Vertex]:
        for edge in self.graph.edge_set:
            if item == edge[0]:
                yield edge[1]
            elif item == edge[1]:
                yield edge[0]


class _DegreeOf(_GraphMapping[int]):
    '''Graph mapping for vertex neibors'''
    def __getitem__(self, item: Vertex) -> int:
        return self.__counter()[item]

    @lru_cache
    def __counter(self) -> tp.Counter[Vertex]:
        a_to_b = Counter(e[0] for e in self.graph.edge_set)
        b_to_a = Counter(e[1] for e in self.graph.edge_set if e[0] != e[1])
        return a_to_b + b_to_a
