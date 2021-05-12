from typing import List, Any, Callable, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_sum_match(list1: List[Any],
                     list2: List[Any],
                     distance_function: Callable[[Any, Any], float]
                     ) -> Tuple[List[Tuple[int, int]],
                                List[int],
                                List[int],
                                Any]:
    matched_idxs: List[Tuple[int, int]] = []
    unmatched_list1_idxs = set(range(len(list1)))
    unmatched_list2_idxs = set(range(len(list2)))
    cost_matrix = np.array([
        [distance_function(item1, item2)
         for item1 in list1]
        for item2 in list2]
    )

    if cost_matrix.ndim == 2:
        matched_list2_idxs, matched_list1_idxs = linear_sum_assignment(
            cost_matrix)

        unmatched_list1_idxs -= set(matched_list1_idxs)
        unmatched_list2_idxs -= set(matched_list2_idxs)
        matched_idxs = list(zip(matched_list1_idxs, matched_list2_idxs))

    return (list(matched_idxs),
            list(unmatched_list1_idxs),
            list(unmatched_list2_idxs),
            cost_matrix)
