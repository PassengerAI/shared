from typing import Tuple

from .BoundingBox import BoundingBox, BoundingBoxError


def pad_to_min_size(bbox: BoundingBox,
                    shape: Tuple[float, float],
                    ) -> BoundingBox:
    """
    Pads `bbox` until it's at least the size of the passed `shape`.
    """
    if bbox.is_empty():
        raise BoundingBoxError('bbox is empty')

    if shape[0] < 0 or shape[1] < 0:
        raise BoundingBoxError('shape is invalid')

    delta_y = shape[0] - bbox.delta_y
    delta_x = shape[1] - bbox.delta_x

    x_min, x_max, y_min, y_max = bbox.x_min, bbox.x_max, bbox.y_min, bbox.y_max

    if delta_y > 0.0:
        y_min -= delta_y / 2
        y_max += delta_y / 2
    if delta_x > 0.0:
        x_min -= delta_x / 2
        x_max += delta_x / 2

    return BoundingBox(x_min, x_max, y_min, y_max)


def pad_to_aspect_ratio(bbox: BoundingBox,
                        aspect_ratio: float
                        ) -> BoundingBox:
    """
    Pads `bbox` to `aspect_ratio` without affecting the other dimension.
    """
    if bbox.is_empty():
        raise BoundingBoxError('bbox is empty')

    if aspect_ratio <= 0.0:
        raise BoundingBoxError('aspect_ratio is invalid')

    height, width = bbox.delta_y, bbox.delta_x

    cur_aspect_ratio = width / height

    x_min, x_max, y_min, y_max = bbox.x_min, bbox.x_max, bbox.y_min, bbox.y_max

    target_height, target_width = height, width

    if aspect_ratio < cur_aspect_ratio:
        target_height = height * cur_aspect_ratio / aspect_ratio
    elif aspect_ratio > cur_aspect_ratio:
        target_width = width * aspect_ratio / cur_aspect_ratio

    return pad_to_min_size(bbox, (target_height, target_width))


def pad_abs_amount(bbox: BoundingBox,
                   shape: Tuple[float, float],
                   ) -> BoundingBox:
    """
    Pads `bbox` by `shape`.
    """
    if bbox.is_empty():
        raise BoundingBoxError('bbox is empty')

    if shape[0] < 0 or shape[1] < 0:
        raise BoundingBoxError('shape is invalid')

    delta_y = shape[0] + bbox.delta_y
    delta_x = shape[1] + bbox.delta_x

    return pad_to_min_size(bbox, (delta_y, delta_x))
