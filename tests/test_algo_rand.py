'''Tests for `hogc.algo.rand` module.'''


from hogc.algo import rand
from collections import Counter


import itertools
import random


def test_rand_uni():
    data = ['a', 'b']
    count = Counter([rand.rand_uni(data) for _ in range(1000)])
    assert count['a'] > 450
    assert count['b'] > 450


def test_rand_in_range():
    count = Counter([rand.rand_in_range(range(1, 3)) for _ in range(1000)])
    assert count[1] > 450
    assert count[2] > 450

    count = Counter(
            [rand.rand_in_range(range(20, 40, 10)) for _ in range(1000)])
    assert count[20] > 450
    assert count[30] > 450
    assert len(count) == 2


def test_rand_threshold():
    count = Counter([rand.rand_threshold(0.5) for _ in range(1000)])
    assert count[True] > 450
    assert count[False] > 450


def test_rand_sample():
    data = ['a', 'b', 'c']
    count = Counter(tuple(sorted(rand.sample(data, k=2))) for _ in range(1000))

    for group in itertools.combinations(data, 2):
        assert count[group] > 290


def test_power_law():
    weights = rand.power_law_distribution(10)
    assert sum(weights) == 1
    for i in range(9):
        assert weights[i] > weights[i + 1]


def test_rand_pl():
    weights = rand.power_law_distribution(2)

    data = ['a', 'b']
    count = Counter([rand.rand_pl(data) for _ in range(1000)])

    assert count['a'] > 1000 * weights[0] - 50
    assert count['b'] > 1000 * weights[1] - 50

    data = [i for i in range(20)]
    weights = rand.power_law_distribution(20)
    count = Counter(rand.rand_pl(data) for i in range(3000))

    for i in range(20):
        assert count[i] > 3000 * weights[i] - 200


def test_rand_norm():
    count = Counter([int(rand.rand_norm(0, 1)) for _ in range(1000)])

    assert 1000 * 1.2 > count[0] > 1000 * 0.6
    assert 1000 * 0.4 > count[1] + count[-1] > 1000 * 0.2

    count = Counter([int(rand.rand_norm(0, 10)) for _ in range(1000)])

    assert 1000 * 0.12 > count[0] > 1000 * 0.06


def test_rand_uni_seed():
    seed = 'seeded_test'
    data = ['a', 'b']
    r = random.Random()

    r.seed(seed)
    count_a = Counter(rand.rand_uni(data, rand=r) for _ in range(1000))

    r.seed(seed)
    count_b = Counter(rand.rand_uni(data, rand=r) for _ in range(1000))

    assert count_a == count_b


def test_rand_in_range_seed():
    seed = 'seeded_test'
    r = random.Random()

    r.seed(seed)
    count_a = Counter(
            rand.rand_in_range(range(1, 999), rand=r) for _ in range(1000))

    r.seed(seed)
    count_b = Counter(
            rand.rand_in_range(range(1, 999), rand=r) for _ in range(1000))

    assert count_a == count_b


def test_rand_sample_seed():
    seed = 'seeded_test'
    data = ['a', 'b', 'c']
    r = random.Random()

    r.seed(seed)
    count_a = Counter(tuple(sorted(rand.sample(data, k=2, rand=r)))
                      for _ in range(1000))

    r.seed(seed)
    count_b = Counter(tuple(sorted(rand.sample(data, k=2, rand=r)))
                      for _ in range(1000))

    assert count_a == count_b


def test_rand_pl_seed():
    seed = 'seeded_test'
    data = ['a', 'b', 'c', 'd']
    r = random.Random()

    r.seed(seed)
    count_a = Counter(rand.rand_pl(data, rand=r) for _ in range(1000))

    r.seed(seed)
    count_b = Counter(rand.rand_pl(data, rand=r) for _ in range(1000))

    assert count_a == count_b


def test_rand_norm_seed():
    seed = 'seeded_test'
    r = random.Random()

    r.seed(seed)
    count_a = Counter([rand.rand_norm(0, 1, rand=r) for _ in range(1000)])
    count_b = Counter([rand.rand_norm(0, 1, rand=r) for _ in range(1000)])

    assert count_a != count_b

    r.seed(seed)
    count_a = Counter([10 * rand.rand_norm(0, 1, rand=r) for _ in range(1000)])
    r.seed(seed)
    count_b = Counter([rand.rand_norm(0, 10, rand=r) for _ in range(1000)])

    assert count_a == count_b
