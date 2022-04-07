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
    return rand.choice(data)


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
    return rand.choices(data, weights=dist(len(data)), k=1)[0]


@__rand_safe
def rand_norm(
            mean: float,
            deviation: float,
            *,
            rand: Random) -> float:
    return rand.normalvariate(mean, deviation)
