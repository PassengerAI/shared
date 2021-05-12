from typing import Any

from hypothesis._strategies import composite, floats

from paitypes.geometry.bounding_box import BoundingBox


@composite
def bounding_boxes(draw: Any, container: BoundingBox = None) -> BoundingBox:
    x_min = -10000
    y_min = -10000
    w_max = 10000
    h_max = 10000
    if container:
        x_min = container.x_min
        y_min = container.y_min
        w_max = container.delta_x
        h_max = container.delta_y

    x = draw(floats(min_value=x_min, max_value=10000.0))
    y = draw(floats(min_value=y_min, max_value=10000.0))
    w = draw(floats(min_value=0.0, max_value=w_max))
    h = draw(floats(min_value=0.0, max_value=h_max))
    return BoundingBox(x, x + w, y, y + h)
