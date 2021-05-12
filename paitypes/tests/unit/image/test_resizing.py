import pytest
import numpy as np

from typing import List, Tuple

from paitypes.geometry.bounding_box import BoundingBox
from paitypes.geometry.Shape import Shape

from paitypes.image import Image, BGRImage, GrayscaleImage
from paitypes.image.resizing import (
    ImageResizingException,
    resize_image_to_size,
    crop_image_to_bounding_box)

from paitypes.tests.fixtures.fixture_image import (
    random_grayscale_image,
    random_bgr_image,
    black_grayscale_image,
    gray_grayscale_image,
    white_grayscale_image,
    grayscale_images,
    black_bgr_image,
    gray_bgr_image,
    white_bgr_image,
    bgr_images,
    all_valid_images,
    empty_bgr_image,
    empty_grayscale_image,
    empty_images)
from paitypes.tests.fixtures.fixture_bounding_box import (
    empty_bbox, full_bbox, partial_bbox, partial_float_bbox)


class TestResizeImageToSize():
    @pytest.mark.parametrize('size', [
        Shape(30, 20),
        Shape(100, 50),
        Shape(100, 100),
        Shape(2000, 5000)
    ])
    def test_resize_image_to_valid_size(self,
                                        all_valid_images: List[Image],
                                        size: Shape
                                        ) -> None:
        for image in all_valid_images:
            resized_image = resize_image_to_size(image, size)
            assert (resized_image.shape[:2] == (size.height, size.width))
            assert (len(resized_image.shape) == len(image.shape))

    @pytest.mark.parametrize('size', [
        Shape(-100, -100),
        Shape(-100, 1),
        Shape(1, -100),
        Shape(0, 0),
        Shape(1, 0),
        Shape(0, 1),
        Shape(-100, 100),
        Shape(100, -100)
    ])
    def test_resize_image_to_invalid_size_raises(
            self,
            all_valid_images: List[Image],
            size: Shape
    ) -> None:
        for image in all_valid_images:
            with pytest.raises(ImageResizingException):
                resize_image_to_size(image, size)

    def test_resize_empty_image_to_size_raises(self,
                                               empty_images: List[Image]
                                               ) -> None:
        for empty_image in empty_images:
            with pytest.raises(ImageResizingException):
                resize_image_to_size(empty_image, Shape(30, 30))


class TestResizeCropImageToBoundingBox():
    def test_crop_to_partial_bbox(self,
                                  all_valid_images: List[Image],
                                  partial_bbox: BoundingBox) -> None:
        for image in all_valid_images:
            cropped_image = crop_image_to_bounding_box(image, partial_bbox)
            assert (np.array_equal(cropped_image, image[26:70, 25:75]))

    def test_crop_to_full_bbox(self,
                               all_valid_images: List[Image],
                               full_bbox: BoundingBox) -> None:
        for image in all_valid_images:
            cropped_image = crop_image_to_bounding_box(image, full_bbox)
            assert (np.array_equal(cropped_image, image))

    def test_crop_to_partial_float_bbox(self,
                                        all_valid_images: List[Image],
                                        partial_float_bbox: BoundingBox
                                        ) -> None:
        for image in all_valid_images:
            cropped_image = crop_image_to_bounding_box(image,
                                                       partial_float_bbox)
            assert (np.array_equal(cropped_image, image[27:74, 33:68]))

    def test_crop_to_empty_bbox_raises(self,
                                       all_valid_images: List[Image],
                                       empty_bbox: BoundingBox) -> None:
        for image in all_valid_images:
            with pytest.raises(ImageResizingException):
                crop_image_to_bounding_box(image, empty_bbox)

    def test_crop_to_empty_image_raises(self,
                                        empty_images: List[Image],
                                        full_bbox: BoundingBox) -> None:
        for empty_image in empty_images:
            with pytest.raises(ImageResizingException):
                crop_image_to_bounding_box(empty_image, full_bbox)

    @pytest.mark.parametrize('bbox', [
        BoundingBox(-1.0, 0.0, 0.0, 100.0),
        BoundingBox(0.0, 0.0, -1.0, 100.0),
        BoundingBox(0.0, 101.0, 0.0, 100.0),
        BoundingBox(0.0, 100.0, 0.0, 101.0)])
    def test_crop_to_out_of_bounds_bbox_raises(self,
                                               all_valid_images: Image,
                                               bbox: BoundingBox
                                               ) -> None:
        for image in all_valid_images:
            with pytest.raises(ImageResizingException):
                crop_image_to_bounding_box(image, bbox)
