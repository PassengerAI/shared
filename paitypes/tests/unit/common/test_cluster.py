from random import shuffle, random, randint, uniform
from typing import List, Set, Dict

import pytest
from hypothesis import given
from hypothesis._strategies import integers, floats, lists
from hypothesis.strategies import sets

from paitypes.common.sequence import cluster, moving_average
from paitypes.geometry.bounding_box import BoundingBox
from paitypes.tests.strategies import bounding_boxes


def euclidean(x: float, y: float) -> float:
    return (x - y) * (x - y)


def add_noise(l: Set[float]) -> Set[float]:
    n = randint(1, 100)
    r = set(l)
    for _ in range(n):
        r.add(random())
    return r


class TestCluster:

    @given(sets(floats(min_value=1e-5, max_value=100.0)))
    def test_map_to_empty(self, set_of_floats: Set[float]) -> None:
        pair_mapping: Dict[float, List[float]] = cluster(euclidean,
                                                         set_of_floats,
                                                         [])
        assert all([pair_mapping[k] == [] for k in pair_mapping])

    @given(sets(floats(min_value=1e-5, max_value=100.0)))
    def test_map_from_empty(self, set_of_floats: Set[float]) -> None:
        pair_mapping: Dict[float, List[float]] = cluster(euclidean,
                                                         set(),
                                                         list(set_of_floats))

        assert pair_mapping == {}

    @given(sets(floats(min_value=1e-5, max_value=100.0)))
    def test_map_to_self(self, set_of_floats: Set[float]) -> None:
        pair_mapping = cluster(euclidean,
                               set_of_floats,
                               list(set_of_floats))

        assert all([pair_mapping[k] == [k] for k in pair_mapping])

    @given(sets(floats(min_value=1e-5, max_value=100.0)))
    def test_include_in_self(self, set_of_floats: Set[float]) -> None:
        second_set = add_noise(set_of_floats)
        pair_mapping = cluster(euclidean,
                               set_of_floats,
                               list(second_set))

        assert all([(k in pair_mapping[k]) for k in pair_mapping])

    @given(sets(floats(min_value=1e-5, max_value=100.0)))
    def test_include_all_if_in_vicinity(self,
                                        set_of_floats: Set[float]) -> None:
        second_set = set()
        for f in set_of_floats:
            second_set.add(f + uniform(0.0, 0.99))

        pair_mapping = cluster(euclidean,
                               set_of_floats,
                               list(second_set))

        total_len = sum([len(l) for _, l in pair_mapping.items()])
        assert total_len == len(second_set)

    @given(sets(floats(min_value=1e-5, max_value=100.0)))
    def test_exclude_all_outside_vicinity(self,
                                          set_of_floats: Set[float]) -> None:
        second_set = set()
        for _ in set_of_floats:
            second_set.add(max(set_of_floats) + uniform(1.1, 10.0))

        pair_mapping = cluster(euclidean,
                               set_of_floats,
                               list(second_set))

        total_len = sum([len(l) for _, l in pair_mapping.items()])
        assert total_len == 0, (pair_mapping, second_set, set_of_floats)

    @given(sets(floats(min_value=1e-5, max_value=100)))
    def test_map_includes_all_keys(self, set_of_floats: Set[float]) -> None:
        second_set = add_noise(set_of_floats)
        pair_mapping = cluster(euclidean,
                               set_of_floats,
                               list(second_set))

        assert len(pair_mapping) == len(set_of_floats)
