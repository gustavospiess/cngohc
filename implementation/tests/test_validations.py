'''tests for confirming the correct behavior of the validations algorithms'''


from hogc import validations
from hogc import models

import pytest


@pytest.mark.parametrize('data,expected', [
    (
        (), 0),
    (
        ((1, 1),), 0),
    (
        ((1, 1), (1, 0)), 0.5),
    (
        ((1, 1), (1, 0), (1, 2)), 2),
    (
        ((2, 2), (2, 0), (2, 4)), 8),
    (
        ((1, 1), (0, 1)), 0.5),
    (
        ((1, 1), (0, 1), (2, 1)), 2),
    (
        ((2, 2), (0, 2), (4, 2)), 8),
    ])
def test_inertia(data, expected):
    assert validations.inertia(
            tuple(models.Vector(d) for d in data)) == expected
