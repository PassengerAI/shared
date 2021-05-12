import cv2
import numpy as np

from typing import TypeVar

from paitypes.image import Image

ImageT = TypeVar('ImageT', bound=Image)


class ImageDifferenceException(TypeError):
    pass


def absolute_image_difference(image1: ImageT,
                              image2: ImageT
                              ) -> ImageT:
    if image1.dtype != image2.dtype:
        raise ImageDifferenceException('image dtypes do not match')

    if image1.shape != image2.shape:
        raise ImageDifferenceException('image dimensions do not match')

    diff = np.abs(
        # Subtracting floats avoids integer under- and overflow.
        image1.astype(np.float64) - image2.astype(np.float64)
    ).astype(image1.dtype)

    return diff
