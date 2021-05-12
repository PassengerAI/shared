from random import random, randint
from typing import List, Set

from hypothesis import given
from hypothesis._strategies import floats, lists

from paitypes.common.optimization import linear_sum_match


def euclidean(x: float, y: float) -> float:
    return (x - y) * (x - y)


def add_noise(l: Set[float]) -> Set[float]:
    n = randint(1, 100)
    r = set(l)
    for _ in range(n):
        r.add(random())
    return r


def test_empty_to_empty() -> None:
    matches, unmatched_1, unmatched_2, _ = linear_sum_match([], [], euclidean)

    assert matches == []
    assert unmatched_1 == []
    assert unmatched_2 == []


@given(lists(floats(min_value=1e-5, max_value=100.0)))
def test_list_to_empty(fs: List[float]) -> None:
    matches, unmatched_1, unmatched_2, _ = linear_sum_match(fs, [], euclidean)

    assert matches == []
    assert unmatched_1 == list(range(len(fs)))
    assert unmatched_2 == []


@given(lists(floats(min_value=1e-5, max_value=100.0)))
def test_list_to_self(fs: List[float]) -> None:
    matches, unmatched_1, unmatched_2, _ = linear_sum_match(fs, fs, euclidean)

    assert matches == list(zip(range(len(fs)), range(len(fs))))
    assert unmatched_1 == []
    assert unmatched_2 == []


@given(lists(floats(min_value=1e-5, max_value=100.0)))
def test_empty_to_list(fs: List[float]) -> None:
    matches, unmatched_1, unmatched_2, _ = linear_sum_match([], fs, euclidean)

    assert matches == []
    assert unmatched_1 == []
    assert unmatched_2 == list(range(len(fs)))
