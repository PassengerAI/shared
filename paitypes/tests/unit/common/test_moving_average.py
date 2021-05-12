from typing import Any, List

import pytest
from hypothesis import given, assume
from hypothesis.strategies import integers
from hypothesis.strategies import floats
from hypothesis.strategies import lists

from paitypes.common.sequence import moving_average


def test_simple_case() -> None:
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]


@given(lists(floats(allow_nan=False)))
def test_window_size_one_is_identity(l: List[float]) -> None:
    assert moving_average(l, 1) == l


@given(lists(floats(allow_nan=False)))
def test_non_positive_window_size_raises(l: List[float]) -> None:
    with pytest.raises(ValueError):
        assert moving_average(l, 0) == l

    with pytest.raises(ValueError):
        assert moving_average(l, -2) == l


@given(integers(min_value=1, max_value=100))
def test_empty_list_returns_empty_list(window_size: int) -> None:
    assert moving_average([], window_size) == []


@given(floats(min_value=-1000, max_value=1000),
       integers(min_value=0, max_value=10),
       integers(min_value=1, max_value=10))
def test_constant_list_returns_constant_ma(value: float,
                                           list_size: int,
                                           window_size: int) -> None:
    ma = moving_average([value] * list_size, window_size)
    assert all([e == value for e in ma])


@given(lists(floats(allow_nan=False), max_size=10),
       integers(min_value=1, max_value=10))
def test_length_when_window_size_lte_list(l: List[float],
                                          window_size: int) -> None:
    assume(window_size <= len(l))
    assert len(moving_average(l, window_size)) == len(l) - window_size + 1


@given(lists(floats(allow_nan=False), max_size=10, min_size=1),
       integers(min_value=11, max_value=100))
def test_length_when_window_size_gt_list(l: List[float],
                                         window_size: int) -> None:
    assert len(moving_average(l, window_size)) == 1
