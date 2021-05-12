import pytest
import numpy as np

from paitypes.tests.fixtures.fixture_image import (
    black_bgr_image,
    gray_bgr_image,
    white_bgr_image,
    white_grayscale_image)

from paitypes.image import Image, BGRImage, GrayscaleImage
from paitypes.image.difference import (ImageDifferenceException,
                                       absolute_image_difference)


class TestImageDifference():
    def test_difference(self,
                        gray_bgr_image: BGRImage,
                        white_bgr_image: BGRImage
                        ) -> None:
        diff = absolute_image_difference(white_bgr_image, gray_bgr_image)
        assert (np.array_equal(diff - 1, gray_bgr_image))

    def test_abs(self,
                 gray_bgr_image: BGRImage,
                 white_bgr_image: BGRImage
                 ) -> None:
        diff = absolute_image_difference(gray_bgr_image, white_bgr_image)
        assert (np.array_equal(diff - 1, gray_bgr_image))

    def test_formats_raise(self,
                           white_bgr_image: BGRImage,
                           white_grayscale_image: GrayscaleImage
                           ) -> None:
        with pytest.raises(ImageDifferenceException):
            absolute_image_difference(white_grayscale_image, white_bgr_image)

    def test_dimensions_raise(self,
                              white_bgr_image: BGRImage
                              ) -> None:
        white_bgr_image_small = white_bgr_image[99, :, :]
        with pytest.raises(ImageDifferenceException):
            absolute_image_difference(white_bgr_image_small, white_bgr_image)

    def test_dtypes_raise(self,
                          white_bgr_image: BGRImage
                          ) -> None:
        white_bgr_image_float = white_bgr_image.astype(np.float64)
        white_bgr_image_uint8 = white_bgr_image.astype(np.uint8)
        with pytest.raises(ImageDifferenceException):
            absolute_image_difference(white_bgr_image_float,
                                      white_bgr_image_uint8)
