'''
This module is a collection of random functions
'''


from random import Random
from functools import wraps, lru_cache as cached

import typing as tp


DEFAULT_RANDOM = Random()
DEFAULT_RANDOM.seed('spiess')


def __rand_safe(func):
    @wraps(func)
    def decorator(*args, rand: tp.Optional[Random] = None, **kwargs):
        if not rand:
            rand = DEFAULT_RANDOM
        return func(*args, rand=rand, **kwargs)
    return decorator


_T = tp.TypeVar('_T')


@__rand_safe
def rand_uni(data: tp.Sequence[_T], *, rand: Random) -> _T:
    '''
    Choose a member of the given set with a uniform distribution.
    It should be equivalent to calling `rand_pl` passing a distribution method
    that gives an even probability distribution for every entry.
    '''
    if len(data) == 1:
        return tuple(data)[0]
    if len(data) == 0:
        raise Exception()
    return rand.sample(data, k=1)[0]


@__rand_safe
def rand_in_range(_range: range, *, rand: Random) -> int:
    return rand.randrange(_range.start, _range.stop, _range.step)


@__rand_safe
def sample(
        data: tp.Union[tp.Sequence[_T], tp.AbstractSet[_T]],
        *,
        k: int,
        rand: Random) -> tp.Set[_T]:
    '''
    Choose a sub set of the given set of length `k` where every member has the
    same probability of being chosen.
    '''
    return set(rand.sample(data, k=k))


Distribution = tp.Callable[[int], tp.Tuple[int, ...]]
'''
The type of a distribution, useful for calling `rand_pl`
'''


@cached
def power_law_distribution(size: int) -> tp.Tuple[int, ...]:
    '''
    Distribution implementation used by Largeron (2015) for power law.

    It gives each member of the sequence the chance:
    frac{x^-2, sum{i=1}{n} i^-2}

    Where `x` is the position of the member in the sequence and `n` is the
    length of the sequence.
    '''
    def prob_density(x):
        return x**-2/sum(i**-2 for i in range(1, size+1))
    weigh = tuple(prob_density(x) for x in range(1, size+1))
    return weigh


@__rand_safe
def rand_pl(
        data: tp.Sequence[_T],
        *,
        dist: Distribution = power_law_distribution,
        rand: Random) -> _T:
    '''
    Returns a member of the given sequence weighed by a power law.
    The default power law used is the `power_law_distribution`, but others can
    be chosen.
    '''
    if len(data) == 1:
        return data[0]
    if len(data) == 0:
        raise ValueError()
    return rand.choices(data, weights=dist(len(data)), k=1)[0]


@__rand_safe
def rand_edge_within(
        data: tp.Set[_T],
        degree: tp.Callable[[_T], int],
        *,
        rand: Random) -> _T:
    '''
    Given a set and a function to map it into a integer named data and degree
    respectively, this function returns a random member of the set with the
    probability distribution equals to the degree of the item over the sum of
    all degrees.
    '''
    t_data = tuple(data)
    if len(t_data) == 1:
        # jus tiniticalized a community with one representative
        return t_data[0]
    total_degree = sum(degree(v) for v in data)
    weights = tuple(degree(v)/total_degree for v in data)
    return rand.choices(t_data, weights=weights, k=1)[0]


@cached
def edge_between_weigths(data, other):
    def distance(vertex, part):
        return part.weighed_distance(vertex, other, 1.0)
    total_degree = sum(1/distance(*v) for v in data)
    weights = tuple(1/distance(*v)/total_degree for v in data)
    return weights


@__rand_safe
def rand_edge_between(data, other, lenth, *, rand: Random):
    '''
    Given a set of pairs of representatives and the partition represented by
    them, an vertex, and the quantity of edges to build, this returns a frozen
    set of representatives to become linket to the provided vertex.
    '''
    if (lenth == 0):
        return frozenset()
    t_data = tuple(data)
    if len(t_data) == 0:
        return frozenset()
    if len(t_data) <= lenth:
        return frozenset(t_data)
    weights = edge_between_weigths(data, other)
    return frozenset(
            p[0] for p in rand.choices(t_data, weights=weights, k=lenth))


@__rand_safe
def rand_norm(
            mean: float,
            deviation: float,
            *,
            rand: Random) -> float:
    '''
    Returns a random float according to a normal distribution, with the given
    mean and deviation.
    '''
    return rand.normalvariate(mean, deviation)


@__rand_safe
def rand_threshold(threshold: float, *, rand: Random) -> bool:
    '''
    Returns a random bool controlled by the threshold.

    For a 75% of the time True, the call would be `rand_threshold(0.75)`
    '''
    return rand.uniform(0, 1) < threshold


@__rand_safe
def shuffle(
        data: tp.List[_T],
        *, rand: Random):
    rand.shuffle(data)
