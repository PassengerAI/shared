import pytest
import numpy as np

from typing import List

from paitypes.image import Image, BGRImage, GrayscaleImage
from paitypes.image.channel import image_channels

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
    bgr_images)


class TestImageChannel:
    def test_grayscale_single_channel(self,
                                      grayscale_images: List[GrayscaleImage]
                                      ) -> None:
        for grayscale_image in grayscale_images:
            channels = image_channels(grayscale_image)
            assert (len(channels) == 1)
            assert (np.array_equal(channels[0], grayscale_image))

    def test_bgr_three_channels(self,
                                bgr_images: List[BGRImage]
                                ) -> None:
        for bgr_image in bgr_images:
            channels = image_channels(bgr_image)
            assert (len(channels) == 3)
            for c in range(3):
                assert (np.array_equal(channels[c], bgr_image[:, :, c]))
