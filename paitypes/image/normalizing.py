import cv2
import numpy as np

from typing import Sequence, TypeVar

from paitypes.image import Image

ImageT = TypeVar('ImageT', bound=Image)


class ImageNormalizingException(IndexError):
    pass


def normalize_image_histogram(image: ImageT) -> ImageT:
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ImageNormalizingException('image size is invalid')

    if image.dtype != np.uint8:
        raise ImageNormalizingException('image dtype must be uint8')

    if len(image.shape) == 2:
        normalized_channels = [cv2.equalizeHist(image)]
    else:
        normalized_channels = [
            cv2.equalizeHist(image[:, :, c]) for c in range(image.shape[2])]

    normalized_image = cv2.merge(normalized_channels)

    return normalized_image.astype(image.dtype)
