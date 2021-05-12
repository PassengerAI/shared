import cv2
import numpy as np

from typing import Tuple, TypeVar, cast

from paitypes.geometry.bounding_box import (BoundingBox, pad_to_min_size,
                                            pad_abs_amount,
                                            pad_to_aspect_ratio, snap_to_shape)
from paitypes.geometry.Shape import Shape
from paitypes.image import Image

ImageT = TypeVar('ImageT', bound=Image)


class ImageResizingException(IndexError):
    pass


def resize_image_to_size(image: ImageT,
                         target_size: Shape,
                         interpolation_method: int = cv2.INTER_AREA
                         ) -> ImageT:
    image_size = Shape.from_image(image)
    if image_size.width <= 0 or image_size.height <= 0:
        raise ImageResizingException('image shape is invalid')
    if target_size.width <= 0 or target_size.height <= 0:
        raise ImageResizingException('target_size is invalid')

    return cv2.resize(image,
                      (target_size.width, target_size.height),
                      interpolation_method)


def crop_image_to_bounding_box(image: ImageT,
                               bbox: BoundingBox
                               ) -> ImageT:
    """
    Crops `image` to the dimensions of `bbox`.
    """
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ImageResizingException('image shape is invalid')

    if bbox.is_empty():
        raise ImageResizingException('bbox is empty')

    y_min, y_max, x_min, x_max = (int(bbox.y_min), int(bbox.y_max),
                                  int(bbox.x_min), int(bbox.x_max))

    if (y_min < 0 or x_min < 0 or
            y_max > image.shape[0] or x_max > image.shape[1]):
        raise ImageResizingException('bbox outside image bounds')

    return image[y_min:y_max, x_min:x_max]


def shape_image_to_bbox(image: np.ndarray,
                        bbox: BoundingBox,
                        thumbnail_size: Tuple[float, float]
                        ) -> np.ndarray:
    # TODO: Change thumbnail size to paitypes.geometry.Shape

    if bbox.is_empty():
        return image

    # TODO: Refactor into `image.resizing` module as a helper method to be
    # shared across detectors.

    # TODO: Use paitypes.geometry.Shape for the following logic
    padded_bbox = pad_to_min_size(bbox, thumbnail_size)
    padded_bbox = pad_abs_amount(padded_bbox, (20.0, 20.0))

    aspect_ratio = (float(thumbnail_size[0]) /
                    float(thumbnail_size[1]))
    padded_bbox = pad_to_aspect_ratio(
        padded_bbox, aspect_ratio)

    # mypy can't figure out that `image.shape[:2]` is limited to two cells
    shape = (image.shape[0], image.shape[1])
    cropped_bbox = snap_to_shape(padded_bbox, shape)

    # TODO: Type this call properly.
    cropped_image = crop_image_to_bounding_box(
        cast(Image, image), cropped_bbox)

    resized_image = cv2.resize(cropped_image,
                               thumbnail_size,
                               interpolation=cv2.INTER_AREA)

    return resized_image
