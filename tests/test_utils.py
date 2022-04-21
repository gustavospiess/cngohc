from hogc import utils
from itertools import chain, count
import pytest


@pytest.mark.parametrize('base', [
    (tuple(range(10)), tuple(range(10))),
    (tuple(range(1, 10, 2)), tuple(range(0, 9, 2))),
    ])
def test_unchain(base):
    data = tuple(zip(*base))
    assert tuple(utils.unchain(chain(*data))) == data


def test_cahced_gen():
    counter = count()

    @utils.cached_generator
    def gen():
        yield next(counter)

    assert tuple(gen()) == tuple(gen())
    next(counter)
    assert tuple(gen()) == tuple(gen())
    assert 0 in tuple(gen())
