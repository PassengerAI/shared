import pytest
import numpy as np

from typing import List, Tuple

from paitypes.geometry.bounding_box import (
    BoundingBox,
    BoundingBoxError,
    pad_to_min_size,
    pad_to_aspect_ratio,
    pad_abs_amount)

from paitypes.tests.fixtures.fixture_bounding_box import (
    empty_bbox, full_bbox, partial_bbox, partial_float_bbox)


class TestResizePadBoundingBoxToMinSize(object):
    @pytest.mark.parametrize('shape', [(100.0, 100.0), (0.0, 0.0)])
    def test_pad_no_padding_returns_original(self,
                                             full_bbox: BoundingBox,
                                             shape: Tuple[float, float]
                                             ) -> None:
        bbox = pad_to_min_size(full_bbox, shape)
        assert (bbox == full_bbox)

    def test_pad_in_center(self,
                           partial_bbox: BoundingBox
                           ) -> None:
        bbox = pad_to_min_size(partial_bbox, (50.0, 60.0))

        assert (bbox.y_min == 23.0)
        assert (bbox.y_max == 73.0)
        assert (bbox.x_min == 20.0)
        assert (bbox.x_max == 80.0)

    def test_pad_out_of_bounds_neg(self) -> None:
        bbox = BoundingBox(0.0, 10.0, 2.0, 12.0)

        padded_bbox = pad_to_min_size(bbox, (20.0, 20.0))

        assert (padded_bbox.y_min == -3.0)
        assert (padded_bbox.y_max == 17.0)
        assert (padded_bbox.x_min == -5.0)
        assert (padded_bbox.x_max == 15.0)

    def test_pad_out_of_bounds_pos(self) -> None:
        bbox = BoundingBox(90.0, 100.0, 92.0, 102.0)

        padded_bbox = pad_to_min_size(bbox, (20.0, 20.0))

        assert (padded_bbox.y_min == 87.0)
        assert (padded_bbox.y_max == 107.0)
        assert (padded_bbox.x_min == 85.0)
        assert (padded_bbox.x_max == 105.0)

    def test_pad_empty_bbox_raises(self,
                                   empty_bbox: BoundingBox
                                   ) -> None:
        with pytest.raises(BoundingBoxError):
            pad_to_min_size(empty_bbox, (20.0, 20.0))

    @pytest.mark.parametrize(
        'shape', [
            (0.0, -1.0), (20.0, -1.0), (-1.0, 20.0), (-10.0, -10.0)])
    def test_pad_invalid_shape_raises(self,
                                      partial_bbox: BoundingBox,
                                      shape: Tuple[float, float]
                                      ) -> None:
        with pytest.raises(BoundingBoxError):
            pad_to_min_size(partial_bbox, shape)


class TestResizePadBoundingBoxToAspectRatio(object):
    def test_pad_same_returns_original(self,
                                       full_bbox: BoundingBox
                                       ) -> None:
        assert (full_bbox ==
                pad_to_aspect_ratio(full_bbox, 100.0 / 100.0))

    def test_pad_along_width(self,
                             full_bbox: BoundingBox
                             ) -> None:
        bbox = pad_to_aspect_ratio(full_bbox, 2.0)
        assert (bbox.y_min == 0.0)
        assert (bbox.y_max == 100.0)
        assert (bbox.x_min == -50.0)
        assert (bbox.x_max == 150.0)

    def test_pad_along_height(self,
                              full_bbox: BoundingBox
                              ) -> None:
        bbox = pad_to_aspect_ratio(full_bbox, 0.5)
        assert (bbox.y_min == -50.0)
        assert (bbox.y_max == 150.0)
        assert (bbox.x_min == 0.0)
        assert (bbox.x_max == 100.0)

    @pytest.mark.parametrize("aspect_ratio", [0.0, 1.0, 2.0])
    def test_pad_empty_bbox_raises(self,
                                   empty_bbox: BoundingBox,
                                   aspect_ratio: float
                                   ) -> None:
        with pytest.raises(BoundingBoxError):
            pad_to_aspect_ratio(empty_bbox, aspect_ratio)

    @pytest.mark.parametrize("aspect_ratio", [0.0, -0.1, -1.0, -2.0])
    def test_pad_invalid_aspect_ratio_raises(self,
                                             full_bbox: BoundingBox,
                                             aspect_ratio: float
                                             ) -> None:
        with pytest.raises(BoundingBoxError):
            pad_to_aspect_ratio(full_bbox, aspect_ratio)


class TestResizePadBoundingBoxAbsAmount(object):
    def test_pad_no_padding_returns_original(self,
                                             full_bbox: BoundingBox
                                             ) -> None:
        bbox = pad_abs_amount(full_bbox, (0.0, 0.0))
        assert (bbox == full_bbox)

    def test_pad_in_center(self,
                           full_bbox: BoundingBox
                           ) -> None:
        bbox = pad_abs_amount(full_bbox, (10.0, 20.0))

        assert (bbox.y_min == -5.0)
        assert (bbox.y_max == 105.0)
        assert (bbox.x_min == -10.0)
        assert (bbox.x_max == 110.0)

    def test_pad_out_of_bounds_neg(self) -> None:
        bbox = BoundingBox(0.0, 10.0, 2.0, 12.0)

        padded_bbox = pad_abs_amount(bbox, (10.0, 10.0))

        assert (padded_bbox.y_min == -3.0)
        assert (padded_bbox.y_max == 17.0)
        assert (padded_bbox.x_min == -5.0)
        assert (padded_bbox.x_max == 15.0)

    def test_pad_out_of_bounds_pos(self) -> None:
        bbox = BoundingBox(90.0, 100.0, 92.0, 102.0)

        padded_bbox = pad_abs_amount(bbox, (10.0, 10.0))

        assert (padded_bbox.y_min == 87.0)
        assert (padded_bbox.y_max == 107.0)
        assert (padded_bbox.x_min == 85.0)
        assert (padded_bbox.x_max == 105.0)

    def test_pad_empty_bbox_raises(self,
                                   empty_bbox: BoundingBox
                                   ) -> None:
        with pytest.raises(BoundingBoxError):
            pad_abs_amount(empty_bbox, (20.0, 20.0))

    @pytest.mark.parametrize('shape', [(0.0, -1.0), (20.0, -1.0),
                                       (-1.0, 20.0), (-10.0, -10.0)])
    def test_pad_invalid_shape_raises(self,
                                      partial_bbox: BoundingBox,
                                      shape: Tuple[float, float]
                                      ) -> None:
        with pytest.raises(BoundingBoxError):
            pad_abs_amount(partial_bbox, shape)
