'''Validation arlgorithm'''

from .models import Vector

import typing as tp


def inertia(data: tp.Sequence[Vector]) -> float:
    center = Vector.avg(data)
    return sum(v**2 for d in data for v in (d-center))
