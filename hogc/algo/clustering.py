'''
This module contains the needed cluestering algorithms and its associated
functions.
'''


from sklearn_extra.cluster import KMedoids as _KMedoids
from hogc.models import Vector, Vertex, Partition

import typing as tp


@tp.runtime_checkable
class Cluster(tp.Protocol, tp.Iterable[Vector], tp.Sized):
    center: Vector
    centroid: Vector
    pass


@tp.runtime_checkable
class ClusterSet(tp.Protocol, tp.Iterable[Cluster]):
    '''
    Protocol for the return of a clustering method
    '''

    centers: tp.Iterable[Vector]
    '''The gravity centers of each cluester'''
    centroids: tp.Iterable[Vector]
    '''The node from each cluster nearest to its gravity center'''
    partition: Partition

    def cap(self, lenght: int) -> 'ClusterSet':
        '''
        define again the cluster set by ignoring the least near members of each
        cluster.

        In this way, every cluster have the same size and the inertia of each
        is optimized.
        '''
        ...


class KMedoidsCluster(tp.Sequence[Vector]):
    __slots__ = ['center', 'centroid', '__inner_tuple']
    center: Vector
    centroid: Vector
    __inner_tuple: tp.Tuple[Vector, ...]

    def __init__(self, k: _KMedoids, i: int, data: tp.Sequence[Vector]):
        self.__inner_tuple = tuple(Vector(d)
                                   for d, lab in zip(data, k.labels_)
                                   if i == lab)
        lenght = len(data[0])
        self.center = Vector(sum(d[axis] for d in self)/(lenght+1)
                             for axis in range(lenght))
        self.centroid = Vector(k.cluster_centers_[i])

    def __len__(self) -> int:
        return len(self.__inner_tuple)

    def __getitem__(self, j: tp.Union[int, slice]):
        return self.__inner_tuple[j]

    def __eq__(self, other) -> bool:
        return all(a == b for a, b in zip(self, other))


class KMedoids(tp.Sequence[Cluster]):
    def __init__(
            self,
            data: tp.Iterable[Vector],
            *,
            n_clusters: int):
        '''
        K medoids adapter
        '''
        data = tuple(data)
        self._k = _KMedoids(n_clusters=n_clusters, random_state=0).fit(data)
        self._data = data
        self.centers = tuple(c.center for c in self)
        self.centroids = tuple(c.centroid for c in self)
        self.partition = Partition()
        for cluster in self:
            sub_partition = Partition(Vertex(v) for v in cluster)
            self.partition.add(sub_partition)

    @property
    def min_len(self) -> int:
        return min(len(c) for c in self)

    def __len__(self) -> int:
        return len(self._k.cluster_centers_)

    def __getitem__(self, index: tp.Union[int, slice]):
        if isinstance(index, slice):
            range_sequence = range(index.start or 0,
                                   index.stop or len(self),
                                   index.step or 1,
                                   )
            return tuple(self[i] for i in range_sequence)
        return KMedoidsCluster(self._k, index, self._data)

    def cap(self, lenght: int) -> 'KMedoids':
        new_data: tp.List[Vector] = list()
        for cluster in self:
            sorted_members = tuple(sorted(cluster,
                                          key=lambda n: n - cluster.center))
            new_data.extend(sorted_members[:lenght])
        return KMedoids(new_data, n_clusters=len(self))
