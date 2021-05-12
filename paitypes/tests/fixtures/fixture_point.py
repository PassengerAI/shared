import pytest

from typing import List, Iterator

from paitypes.geometry.Point import Point


@pytest.fixture()
def top_left_point() -> Iterator[Point]:
    yield Point(0.0, 0.0)


@pytest.fixture()
def top_right_point() -> Iterator[Point]:
    yield Point(100.0, 0.0)


@pytest.fixture()
def bot_left_point() -> Iterator[Point]:
    yield Point(0.0, 100.0)


@pytest.fixture()
def bot_right_point() -> Iterator[Point]:
    yield Point(100.0, 100.0)


@pytest.fixture()
def neg_point() -> Iterator[Point]:
    yield Point(-10.0, -10.0)


@pytest.fixture()
def full_bbox_points(top_left_point: Point,
                     top_right_point: Point,
                     bot_left_point: Point,
                     bot_right_point: Point) -> Iterator[List[Point]]:
    yield [top_left_point, top_right_point, bot_left_point, bot_right_point]
