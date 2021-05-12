import pytest

from paitypes.geometry.bounding_box.BoundingBox import BoundingBox


@pytest.fixture()
def empty_bbox() -> BoundingBox:
    yield BoundingBox(0.0, 0.0, 0.0, 0.0)


@pytest.fixture()
def full_bbox() -> BoundingBox:
    yield BoundingBox(0.0, 100.0, 0.0, 100.0)


@pytest.fixture()
def partial_bbox() -> BoundingBox:
    yield BoundingBox(25.0, 75.0, 26.0, 70.0)


@pytest.fixture()
def partial_float_bbox() -> BoundingBox:
    yield BoundingBox(33.3, 68.2, 27.8, 74.5)
