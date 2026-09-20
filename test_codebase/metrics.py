"""Statistical calculations used by the sample project."""

import math
from collections.abc import Sequence


def mean(numbers: Sequence[float]) -> float:
    """Return the arithmetic mean of a non-empty sequence."""
    if not numbers:
        raise ValueError("mean requires at least one number")
    return sum(numbers) / len(numbers)


def variance(numbers: Sequence[float]) -> float:
    """Return population variance, reusing :func:`mean`."""
    average = mean(numbers)
    return sum((number - average) ** 2 for number in numbers) / len(numbers)


def standard_deviation(numbers: Sequence[float]) -> float:
    """Return population standard deviation."""
    if not numbers:
        raise ValueError("standard_deviation requires at least one number")
    return math.sqrt(variance(numbers))
