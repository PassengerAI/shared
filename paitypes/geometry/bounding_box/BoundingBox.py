import math
import numpy as np
from ..Shape import Shape
from ..Point import Point
from paitypes.common.DataError import DataError
from typing import List, Dict
from dataclasses import dataclass


class BoundingBoxError(DataError):
    pass


@dataclass
class BoundingBox:
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0

    @property
    def delta_x(self) -> float:
        return self.x_max - self.x_min

    @property
    def delta_y(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.delta_x * self.delta_y

    @property
    def center(self) -> Point:
        return Point((self.x_min + self.x_max) / 2.0,
                     (self.y_min + self.y_max) / 2.0)

    def is_empty(self) -> bool:
        return self.delta_x <= 0.0 or self.delta_y <= 0.0

    def is_small(self) -> bool:
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min

        area = height * width

        return area < 25000.0

    @classmethod
    def from_dict(cls, datadict: Dict) -> 'BoundingBox':
        if set(datadict.keys()) != {'x1', 'x2', 'y1', 'y2'}:
            raise BoundingBoxError('more/not all keys present')

        try:
            x_min, x_max = float(datadict['x1']), float(datadict['x2'])
            y_min, y_max = float(datadict['y1']), float(datadict['y2'])
        except ValueError:
            raise BoundingBoxError('unparsable input dict')
        return cls(x_min, x_max, y_min, y_max)

    def to_dict(self) -> Dict:
        return {'x1': self.x_min, 'x2': self.x_max,
                'y1': self.y_min, 'y2': self.y_max}

    @classmethod
    def from_tfpose_dict(cls, datadict: Dict[str, int]) -> 'BoundingBox':
        # TfPose returns a dictionary of the form below, with x and y being the
        # center points of the bounding box
        if set(datadict.keys()) != {'x', 'y', 'w', 'h'}:
            raise BoundingBoxError('more/not all keys present')

        try:
            cx, cy = float(datadict['x']), float(datadict['y'])
            w, h = float(datadict['w']), float(datadict['h'])
            if w < 0 or h < 0:
                raise BoundingBoxError(f"Width and height must be positive: "
                                       f"({w}, {h})")
        except ValueError:
            raise BoundingBoxError('unparsable input dict')
        return BoundingBox(cx - w / 2.0,
                           cx + w / 2.0,
                           cy - h / 2.0,
                           cy + h / 2.0)

    @classmethod
    def from_points(cls, points: List[Point]) -> 'BoundingBox':
        xs = list(map(lambda point: point.x, points))
        ys = list(map(lambda point: point.y, points))

        x_min, x_max = min(xs, default=0.0), max(xs, default=0.0)
        y_min, y_max = min(ys, default=0.0), max(ys, default=0.0)

        return cls(x_min, x_max, y_min, y_max)

    @classmethod
    def distance(cls,
                 bounds_one: 'BoundingBox',
                 bounds_two: 'BoundingBox'
                 ) -> float:
        c1_x = (float(bounds_one.x_min) * 0.5) + \
               (float(bounds_one.x_max) * 0.5)
        c1_y = (float(bounds_one.y_min) * 0.5) + \
               (float(bounds_one.y_max) * 0.5)

        c2_x = (float(bounds_two.x_min) * 0.5) + \
               (float(bounds_two.x_max) * 0.5)
        c2_y = (float(bounds_two.y_min) * 0.5) + \
               (float(bounds_two.y_max) * 0.5)

        return math.sqrt(math.pow(c1_x - c2_x, 2) + math.pow(c1_y - c2_y, 2))

    @classmethod
    def mean_bound(cls,
                   bounding_boxes: List['BoundingBox']
                   ) -> 'BoundingBox':
        x_mins = [bbox.x_min for bbox in bounding_boxes]
        y_mins = [bbox.y_min for bbox in bounding_boxes]
        x_maxs = [bbox.x_max for bbox in bounding_boxes]
        y_maxs = [bbox.y_max for bbox in bounding_boxes]
        bound_dict = {
            'x1': np.mean(x_mins),
            'y1': np.mean(y_mins),
            'x2': np.mean(x_maxs),
            'y2': np.mean(y_maxs)
        }

        return cls.from_dict(bound_dict)

    def scale(self,
              shape: Shape,
              center: Point = Point(0.0, 0.0)) -> 'BoundingBox':
        moved_bbox = self.move(-center.x, -center.y)
        centered_bbox = BoundingBox(x_min=moved_bbox.x_min * shape.width,
                                    x_max=moved_bbox.x_max * shape.width,
                                    y_min=moved_bbox.y_min * shape.height,
                                    y_max=moved_bbox.y_max * shape.height)
        return centered_bbox.move(center.x, center.y)

    def to_shape(self) -> Shape:
        return Shape(self.delta_x, self.delta_y)

    def encloses(self, point: Point, fuzz_factor: float = 0.1) -> bool:
        return (
                self.x_min - fuzz_factor <= point.x <= self.x_max + fuzz_factor
                and
                self.y_min - fuzz_factor <= point.y <= self.y_max + fuzz_factor
        )

    def is_inside(self, container: 'BoundingBox') -> bool:
        return (container.x_min <= self.x_min <= container.x_max and
                container.x_min <= self.x_max <= container.x_max and
                container.y_min <= self.y_min <= container.y_max and
                container.y_min <= self.y_max <= container.y_max
                )

    def __str__(self) -> str:
        return '(BoundingBox: x_min={} x_max={} y_min={} y_max={})'.format(
            self.x_min, self.x_max, self.y_min, self.y_max)

    def __add__(self, other: object) -> 'BoundingBox':
        if isinstance(other, BoundingBox):
            if self.is_empty() or other.is_empty():
                raise BoundingBoxError('bbox is empty')

            x_min, x_max = min(
                self.x_min, other.x_min), max(
                self.x_max, other.x_max)
            y_min, y_max = min(
                self.y_min, other.y_min), max(
                self.y_max, other.y_max)
            return BoundingBox(x_min, x_max, y_min, y_max)

        raise NotImplementedError

    def move(self, dx: float, dy: float) -> 'BoundingBox':
        return BoundingBox(self.x_min + dx,
                           self.x_max + dx,
                           self.y_min + dy,
                           self.y_max + dy)

    def __hash__(self) -> int:
        return hash((self.x_min, self.x_max, self.y_min, self.y_max))


def intersection(bbox1: BoundingBox,
                 bbox2: BoundingBox) -> BoundingBox:
    """
        Calculate the intersection of two bounding boxes.
    """

    assert bbox1.x_min <= bbox1.x_max
    assert bbox1.y_min <= bbox1.y_max
    assert bbox2.x_min <= bbox2.x_max
    assert bbox2.y_min <= bbox2.y_max

    # determine the coordinates of the intersection rectangle
    x_left = max(bbox1.x_min, bbox2.x_min)
    y_top = max(bbox1.y_min, bbox2.y_min)
    x_right = min(bbox1.x_max, bbox2.x_max)
    y_bottom = min(bbox1.y_max, bbox2.y_max)
    if x_right < x_left or y_bottom < y_top:
        return EMPTY_BBOX
    return BoundingBox(x_left, x_right, y_top, y_bottom)


def contains_ratio(container: BoundingBox, contained: BoundingBox) -> float:
    """ Returns a float between 0.0 and 1.0 indicating the degree to
    which a bounding box in contained into another one. 0.0  """
    if contained.area == 0:
        return 0.0
    return intersection(container, contained).area / contained.area


def get_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    intersect = intersection(bbox1, bbox2)
    union_area = float(bbox1.area + bbox2.area - intersect.area)
    if union_area == 0.0:
        return 0.0

    # compute the intersection over union_area by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = (intersect.area / union_area)
    assert 0.0 <= iou <= 1.0
    return iou


EMPTY_BBOX = BoundingBox(0.0, 0.0, 0.0, 0.0)
