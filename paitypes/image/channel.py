import cv2
import numpy as np

from typing import List

from . import Image


def image_channels(image: Image) -> List[Image]:
    if len(image.shape) > 2:
        return [image[:, :, c] for c in range(image.shape[2])]
    else:
        return [image]


def to_channel_first(image: Image) -> np.ndarray:
    return image.transpose((2, 0, 1))
