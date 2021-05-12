from typing import Tuple

from .BoundingBox import BoundingBox, BoundingBoxError


def snap_to_shape(bbox: BoundingBox,
                  shape: Tuple[float, float],
                  ) -> BoundingBox:
    """
    Snaps `bbox` into the bounds of y=(0.0, shape[0]), x=(0.0, shape[1]).

    First translates `bbox` into the bounds, then caps the dimensions such that
    the `bbox` fits within `shape`.
    """
    if bbox.is_empty():
        raise BoundingBoxError('bbox is empty')

    if shape[0] < 0 or shape[1] < 0:
        raise BoundingBoxError('shape is invalid')

    x_min, x_max, y_min, y_max = bbox.x_min, bbox.x_max, bbox.y_min, bbox.y_max

    if y_min < 0.0:
        y_max += -y_min
        y_min = 0.0
    if x_min < 0.0:
        x_max += -x_min
        x_min = 0.0

    if y_max > shape[0]:
        y_min -= y_max - shape[0]
        y_max = shape[0]
    if x_max > shape[1]:
        x_min -= x_max - shape[1]
        x_max = shape[1]

    y_min, y_max = max(y_min, 0.0), min(y_max, shape[0])
    x_min, x_max = max(x_min, 0.0), min(x_max, shape[1])

    return BoundingBox(x_min, x_max, y_min, y_max)
