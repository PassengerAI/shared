import cv2
import numpy as np

from typing import NewType, TypeVar, cast, Union


class ImageColorException(ValueError):
    pass


BGRImage = NewType('BGRImage', np.ndarray)
RGBImage = NewType('RGBImage', np.ndarray)
YCrCbImage = NewType('YCrCbImage', np.ndarray)
GrayscaleImage = NewType('GrayscaleImage', np.ndarray)

Image = Union[BGRImage, RGBImage, YCrCbImage, GrayscaleImage]


# When you want to enforce that multiple parameters passed into a function are
# of the same nominal image type, you can create a binding as follows:
# ImageT = TypeVar('ImageT', bound=Image)


def ndarray_to_bgr_image(image: np.ndarray) -> BGRImage:
    if not len(image.shape) == 3:
        raise ImageColorException('image does not have 3 dimensions')
    if image.shape[2] != 3:
        raise ImageColorException('image is not 3 channels')
    return BGRImage(image)


def ndarray_to_grayscale_image(image: np.ndarray) -> GrayscaleImage:
    if not len(image.shape) == 2:
        raise ImageColorException('image does not have 2 dimensions')
    return GrayscaleImage(image)


def bgr_image_to_rgb(bgr_image: BGRImage) -> RGBImage:
    return RGBImage(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))


def rgb_image_to_bgr(rgb_image: RGBImage) -> BGRImage:
    return BGRImage(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))


def bgr_image_to_grayscale(bgr_image: BGRImage) -> GrayscaleImage:
    return GrayscaleImage(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY))


def bgr_image_to_ycrcb(bgr_image: BGRImage) -> YCrCbImage:
    return YCrCbImage(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb))
