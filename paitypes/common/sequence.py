from logging import getLogger
from statistics import mean
from typing import Callable, Iterable, TypeVar, Tuple, Dict, List, Set, \
    Optional

import numpy as np

logger = getLogger(__name__)
T_a = TypeVar('T_a')
T_b = TypeVar('T_b')


def cluster(distance_function: Callable[[T_b, T_a], float],
            clusters: Set[T_a],
            values: List[T_b],
            max_distance: float = 1.0
            ) -> Dict[T_a, List[T_b]]:
    """Put `values` into clusters defined by `clusters`.

    Clustering of `values` is based on the distance of each value to the
    cluster, as calculated by the distance function. Each value is assigned
    to the cluster it is closest to. If a value has similar distance to
    multiple clusters, it is assigned to the cluster that appears earlier in
    `clusters`.

    * The distance function can be asymmetric.

    * If there no cluster with distance less than `max_distance` to a value,
    that value would not be clustered.

    """
    min_distance = {b: max_distance for b in values}
    centers: Dict[T_b, Optional[T_a]] = {b: None for b in values}

    for b in values:
        for a in clusters:
            d = distance_function(b, a)
            if d < min_distance[b]:
                min_distance[b] = d
                centers[b] = a

    cluster_mappings: Dict[T_a, List[T_b]] = {a: [] for a in clusters}
    for b in values:
        center = centers[b]
        if center is not None:
            cluster_mappings[center].append(b)

    return cluster_mappings


def moving_window(f: Callable, l: List, window_size: int) -> List:
    if window_size <= 0:
        raise ValueError("`window_size` must be a positive non-zero integer")
    if not l:
        return []
    window_size = min(window_size, len(l))
    result = []
    for i in range(len(l) - window_size + 1):
        result.append(f(l[i:i + window_size]))
    return result


def moving_average(l: List[float], window_size: int) -> List[float]:
    return moving_window(mean, l, window_size)
