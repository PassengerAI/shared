import pytest
import numpy as np

from typing import List, Tuple

from dataclasses import replace

from paitypes.geometry.bounding_box import BoundingBox, snap_to_shape

from paitypes.tests.fixtures.fixture_bounding_box import (
    empty_bbox, full_bbox, partial_bbox, partial_float_bbox)


class TestResizeSnapBoundingBoxToShape(object):
    @pytest.mark.parametrize('shape', [(100.0, 100.0), (200.0, 200.0)])
    def test_snap_to_larger_returns_original(self,
                                             full_bbox: BoundingBox,
                                             shape: Tuple[float, float]
                                             ) -> None:
        bbox = snap_to_shape(full_bbox, shape)
        assert (bbox == full_bbox)

    def test_snap_to_smaller_returns_snap(self,
                                          full_bbox: BoundingBox
                                          ) -> None:
        bbox = replace(full_bbox, y_min=-10.0, x_min=-10.0)

        snapped_bbox = snap_to_shape(bbox, (50.0, 50.0))

        assert (snapped_bbox.y_min == 0.0)
        assert (snapped_bbox.y_max == 50.0)
        assert (snapped_bbox.x_min == 0.0)
        assert (snapped_bbox.x_max == 50.0)

    def test_snap_at_top(self,
                         partial_bbox: BoundingBox
                         ) -> None:
        bbox = replace(partial_bbox, y_min=-6.0, y_max=44.0)

        snapped_bbox = snap_to_shape(bbox, (100.0, 100.0))

        assert (snapped_bbox.y_min == 0.0)
        assert (snapped_bbox.y_max == 50.0)
        assert (snapped_bbox.x_min == 25.0)
        assert (snapped_bbox.x_max == 75.0)

    def test_snap_at_bottom(self,
                            partial_bbox: BoundingBox
                            ) -> None:
        bbox = replace(partial_bbox, y_min=56.0, y_max=106.0)

        snapped_bbox = snap_to_shape(bbox, (100.0, 100.0))

        assert (snapped_bbox.y_min == 50.0)
        assert (snapped_bbox.y_max == 100.0)
        assert (snapped_bbox.x_min == 25.0)
        assert (snapped_bbox.x_max == 75.0)

    def test_snap_at_left(self,
                          partial_bbox: BoundingBox
                          ) -> None:
        bbox = replace(partial_bbox, x_min=-5.0, x_max=45.0)

        snapped_bbox = snap_to_shape(bbox, (100.0, 100.0))

        assert (snapped_bbox.y_min == 26.0)
        assert (snapped_bbox.y_max == 70.0)
        assert (snapped_bbox.x_min == 0.0)
        assert (snapped_bbox.x_max == 50.0)

    def test_snap_at_right(self,
                           partial_bbox: BoundingBox
                           ) -> None:
        bbox = replace(partial_bbox, x_min=55.0, x_max=105.0)

        snapped_bbox = snap_to_shape(bbox, (100.0, 100.0))

        assert (snapped_bbox.y_min == 26.0)
        assert (snapped_bbox.y_max == 70.0)
        assert (snapped_bbox.x_min == 50.0)
        assert (snapped_bbox.x_max == 100.0)
