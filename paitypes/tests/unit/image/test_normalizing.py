import pytest
import numpy as np

from typing import List

from paitypes.image import Image
from paitypes.image.normalizing import (normalize_image_histogram,
                                        ImageNormalizingException)

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


class TestNormalizingImage():
    def test_respects_num_channels(self,
                                   all_valid_images: List[Image]
                                   ) -> None:
        for image in all_valid_images:
            assert (normalize_image_histogram(image).shape == image.shape)

    @pytest.mark.parametrize('dtype', [np.bool, np.float32, np.float64])
    def test_invalid_type_raises(self,
                                 all_valid_images: List[Image],
                                 dtype: np.dtype
                                 ) -> None:
        for image in all_valid_images:
            image_cast = image.astype(dtype)
            with pytest.raises(ImageNormalizingException):
                normalize_image_histogram(image_cast)

    def test_invalid_shape_raises(self,
                                  empty_images: List[Image]
                                  ) -> None:
        for empty_image in empty_images:
            with pytest.raises(ImageNormalizingException):
                normalize_image_histogram(empty_image)
