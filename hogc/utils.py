'''
TODO: doc
'''


from functools import lru_cache, wraps


import typing as tp
_T = tp.TypeVar('_T')


def unchain(
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
