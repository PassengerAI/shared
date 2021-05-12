from hypothesis._strategies import floats
from math import isclose, sqrt

import pytest

from typing import List, Dict

from dataclasses import replace
from hypothesis import given, assume
from pytest import approx

from paitypes.geometry.Shape import Shape
from paitypes.geometry.bounding_box.BoundingBox import (contains_ratio,
                                                        intersection)
from paitypes.geometry.bounding_box import (BoundingBox,
                                            BoundingBoxError,
                                            EMPTY_BBOX)
from paitypes.geometry.Point import Point
from paitypes.tests.fixtures.fixture_bounding_box import (
    empty_bbox, full_bbox, partial_bbox, partial_float_bbox)
from paitypes.tests.fixtures.fixture_point import (
    top_left_point, top_right_point, bot_left_point, bot_right_point,
    neg_point, full_bbox_points)
from paitypes.tests.strategies import bounding_boxes


def test_is_empty_partial_bbox(partial_bbox: BoundingBox) -> None:
    assert (not partial_bbox.is_empty())


def test_is_empty_full_bbox(full_bbox: BoundingBox) -> None:
    assert (not full_bbox.is_empty())


def test_is_empty_empty_bbox(empty_bbox: BoundingBox) -> None:
    assert (empty_bbox.is_empty())


def test_inverted_bbox_is_empty() -> None:
    bbox = BoundingBox(10.0, 10.0, 0.0, 0.0)
    assert (bbox.is_empty())


@pytest.mark.parametrize('raw_dict', [
    {'x1': 1, 'x2': 2, 'y1': 3, 'y2': 4},
    {'x1': 1.0, 'x2': 2.0, 'y1': 3.0, 'y2': 4.0},
    {'x1': '1', 'x2': '2', 'y1': '3', 'y2': '4'},
    {'x1': '1.0', 'x2': '2.0', 'y1': '3.0', 'y2': '4.0'},
])
def test_from_dict_input_format(raw_dict: Dict) -> None:
    bbox = BoundingBox.from_dict(raw_dict)
    assert (not bbox.is_empty())
    assert (bbox.x_min == 1.0)
    assert (bbox.x_max == 2.0)
    assert (bbox.y_min == 3.0)
    assert (bbox.y_max == 4.0)


@pytest.mark.parametrize('raw_dict', [
    {'x1': 1, 'x2': 2, 'y1': 3, 'y2': 4},
    {'x1': 1.0, 'x2': 2.0, 'y1': 3.0, 'y2': 4.0},
    {'x1': '1', 'x2': '2', 'y1': '3', 'y2': '4'},
    {'x1': '1.0', 'x2': '2.0', 'y1': '3.0', 'y2': '4.0'},
])
def test_from_dict_to_dict(raw_dict: Dict) -> None:
    bbox1 = BoundingBox.from_dict(raw_dict)
    bbox2 = BoundingBox.from_dict(
        BoundingBox.from_dict(raw_dict).to_dict())
    assert (bbox1 == bbox2)


def test_to_dict_from_dict(empty_bbox: BoundingBox,
                           partial_bbox: BoundingBox,
                           full_bbox: BoundingBox) -> None:
    for bbox in [empty_bbox, partial_bbox, full_bbox]:
        assert BoundingBox.from_dict(bbox.to_dict()) == bbox


@pytest.mark.parametrize('raw_dict', [
    {'x1': 'a', 'x2': 'dasdf', 'y1': '3', 'y2': '4'},
    {'x1': '1.0', 'x2': '2.0', 'y1': 'fjh', 'y2': '4.0'},
    {'x2': '2.0', 'y1': '3.0', 'y2': '4.0'},
    {'x1': '1.0', 'x2': '2.0', 'y1': '3.0'},
    {'x1': '1.0', 'x2': '2.0', 'y1': '3.0', 'y2': '4.0', 'y3': '5.0'}
])
def test_from_dict_invalid_input(raw_dict: Dict) -> None:
    with pytest.raises(BoundingBoxError):
        bbox = BoundingBox.from_dict(raw_dict)


def test_from_points_valid(full_bbox_points: List[Point],
                           full_bbox: BoundingBox) -> None:
    bbox = BoundingBox.from_points(full_bbox_points)
    assert (not bbox.is_empty())
    assert (bbox == full_bbox)


def test_from_points_valid_neg(full_bbox_points: List[Point],
                               neg_point: Point,
                               full_bbox: BoundingBox) -> None:
    points = full_bbox_points + [neg_point]
    bbox = BoundingBox.from_points(points)
    expected_bbox = replace(full_bbox, x_min=-10.0, y_min=-10.0)
    assert (not bbox.is_empty())
    assert (bbox == expected_bbox)


def test_from_single_point_empty(full_bbox_points: List[Point]) -> None:
    for point in full_bbox_points:
        bbox = BoundingBox.from_points([point])
        assert (bbox.is_empty())


def test_eq_is_equal(empty_bbox: BoundingBox,
                     partial_bbox: BoundingBox,
                     full_bbox: BoundingBox) -> None:
    for a, b in [
        (empty_bbox, BoundingBox(0.0, 0.0, 0.0, 0.0)),
        (partial_bbox, BoundingBox(25.0, 75.0, 26.0, 70.0)),
        (full_bbox, BoundingBox(0.0, 100.0, 0.0, 100.0)),
    ]:
        assert (a == b)


def test_eq_not_equal(empty_bbox: BoundingBox,
                      partial_bbox: BoundingBox,
                      full_bbox: BoundingBox) -> None:
    for a, b in [
        (empty_bbox, partial_bbox),
        (partial_bbox, full_bbox),
        (full_bbox, empty_bbox),
    ]:
        assert (a != b)


def test_eq_invalid_type(empty_bbox: BoundingBox,
                         partial_bbox: BoundingBox,
                         full_bbox: BoundingBox) -> None:
    for a, b in [
        (empty_bbox, 3),
        (partial_bbox, 'a'),
        (full_bbox, Point(5, 8)),
    ]:
        assert (a != b)


def test_add_set_and_superset_returns_superset(full_bbox: BoundingBox,
                                               empty_bbox: BoundingBox,
                                               partial_bbox: BoundingBox
                                               ) -> None:
    for a, b in [
        (partial_bbox, partial_bbox),
        (partial_bbox, full_bbox),
        (full_bbox, full_bbox)
    ]:
        assert (a + b == b)


def test_offset_bboxes_returns_union(full_bbox: BoundingBox) -> None:
    other = BoundingBox(-100.0, 0.0, -100.0, 0.0)
    expected = BoundingBox(-100.0, 100.0, -100.0, 100.0)
    assert (full_bbox + other == expected)


def test_add_empty_bbox_raises(full_bbox: BoundingBox,
                               empty_bbox: BoundingBox,
                               partial_bbox: BoundingBox) -> None:
    for a, b in [
        (empty_bbox, empty_bbox),
        (empty_bbox, partial_bbox),
        (partial_bbox, empty_bbox),
        (empty_bbox, full_bbox)
    ]:
        with pytest.raises(BoundingBoxError):
            a + b


def test_add_other_type_raises(full_bbox: BoundingBox,
                               partial_bbox: BoundingBox) -> None:
    for a, b in [
        (partial_bbox, 3),
        (partial_bbox, 'a'),
        (full_bbox, Point(0.0, 3.0))
    ]:
        with pytest.raises(NotImplementedError):
            a + b


@given(bounding_boxes())
def test_intersection_same(bbox1: BoundingBox) -> None:
    assert intersection(bbox1, bbox1) == bbox1


@given(bounding_boxes())
def test_intersection_empty(empty_bbox: BoundingBox,
                            bbox1: BoundingBox) -> None:
    assert intersection(bbox1, empty_bbox) == empty_bbox


@given(bounding_boxes(), bounding_boxes())
def test_intersection_symmetric(bbox1: BoundingBox,
                                bbox2: BoundingBox) -> None:
    assert intersection(bbox1, bbox2) == intersection(bbox2, bbox1)


@given(bounding_boxes())
def test_fully_contains(bbox1: BoundingBox) -> None:
    contained = BoundingBox(bbox1.x_min, bbox1.x_min + bbox1.delta_x / 2.0,
                            bbox1.y_min, bbox1.y_min + bbox1.delta_y / 2.0)

    assume(contained.area != 0.0)
    assert contains_ratio(bbox1, contained) == 1.0


@given(bounding_boxes())
def test_partially_contains(bbox1: BoundingBox) -> None:
    moved_bbox = BoundingBox(bbox1.x_min - bbox1.delta_x / 2.0,
                             bbox1.x_min + bbox1.delta_x / 2.0,
                             bbox1.y_min - bbox1.delta_y / 2.0,
                             bbox1.y_min + bbox1.delta_y / 2.0)

    assume(moved_bbox.area != 0.0)
    assert isclose(contains_ratio(bbox1, moved_bbox), 0.25,
                   abs_tol=1e-5)


@given(bounding_boxes(), bounding_boxes())
def test_intersection_contained(bbox1: BoundingBox,
                                bbox2: BoundingBox) -> None:
    if contains_ratio(bbox1, bbox2) == 1.0:
        assert intersection(bbox1, bbox2) == bbox2


@given(bounding_boxes(), bounding_boxes())
def test_intersection_not_contained(bbox1: BoundingBox,
                                    bbox2: BoundingBox) -> None:
    if contains_ratio(bbox1, bbox2) == 0.0:
        assert (intersection(bbox1, bbox2) != bbox2 or
                bbox2.area == 0.0 or
                bbox1.area == 0.0
                )


@given(bounding_boxes())
def test_no_move(bbox: BoundingBox) -> None:
    assert bbox.move(0, 0) == bbox


@given(bounding_boxes(),
       floats(min_value=-10.0, max_value=10.0),
       floats(min_value=-10.0, max_value=10.0))
def test_scale_around_origin_changes_size(bbox: BoundingBox,
                                          sx: float,
                                          sy: float) -> None:
    scaled_bbox = bbox.scale(Shape(sx, sy))
    assert scaled_bbox.delta_x == approx(bbox.delta_x * sx, abs=1e-6)
    assert scaled_bbox.delta_y == approx(bbox.delta_y * sy, abs=1e-6)


@given(bounding_boxes(),
       floats(min_value=-10.0, max_value=10.0),
       floats(min_value=-10.0, max_value=10.0))
def test_scale_around_origin_moves_center(bbox: BoundingBox,
                                          sx: float,
                                          sy: float) -> None:
    scaled_bbox = bbox.scale(Shape(sx, sy))
    assert scaled_bbox.center.x == approx(bbox.center.x * sx, abs=1e-6)
    assert scaled_bbox.center.y == approx(bbox.center.y * sy, abs=1e-6)


@given(bounding_boxes(),
       floats(min_value=-10.0, max_value=10.0),
       floats(min_value=-10.0, max_value=10.0))
def test_scale_around_center_changes_size(bbox: BoundingBox,
                                          sx: float,
                                          sy: float) -> None:
    scaled_bbox = bbox.scale(Shape(sx, sy), center=bbox.center)
    assert scaled_bbox.delta_x == approx(bbox.delta_x * sx, abs=1e-6)
    assert scaled_bbox.delta_y == approx(bbox.delta_y * sy, abs=1e-6)


@given(bounding_boxes(),
       floats(min_value=-10.0, max_value=10.0),
       floats(min_value=-10.0, max_value=10.0))
def test_scale_around_center_maintains_center(bbox: BoundingBox,
                                              sx: float,
                                              sy: float) -> None:
    scaled_bbox = bbox.scale(Shape(sx, sy), center=bbox.center)
    assert scaled_bbox.center.x == approx(bbox.center.x, abs=1e-6)
    assert scaled_bbox.center.y == approx(bbox.center.y, abs=1e-6)


@given(bounding_boxes(),
       floats(min_value=-10.0, max_value=10.0),
       floats(min_value=-10.0, max_value=10.0))
def test_scale_around_corners_maintains_corners(bbox: BoundingBox,
                                                sx: float,
                                                sy: float) -> None:
    scaled_bbox = bbox.scale(Shape(sx, sy),
                             center=Point(bbox.x_min, bbox.y_min))
    assert scaled_bbox.x_min == approx(bbox.x_min, abs=1e-6)
    assert scaled_bbox.y_min == approx(bbox.y_min, abs=1e-6)

    scaled_bbox = bbox.scale(Shape(sx, sy),
                             center=Point(bbox.x_min, bbox.y_max))
    assert scaled_bbox.x_min == approx(bbox.x_min, abs=1e-6)
    assert scaled_bbox.y_max == approx(bbox.y_max, abs=1e-6)

    scaled_bbox = bbox.scale(Shape(sx, sy),
                             center=Point(bbox.x_max, bbox.y_min))
    assert scaled_bbox.x_max == approx(bbox.x_max, abs=1e-6)
    assert scaled_bbox.y_min == approx(bbox.y_min, abs=1e-6)

    scaled_bbox = bbox.scale(Shape(sx, sy),
                             center=Point(bbox.x_max, bbox.y_max))
    assert scaled_bbox.x_max == approx(bbox.x_max, abs=1e-6)
    assert scaled_bbox.y_max == approx(bbox.y_max, abs=1e-6)
