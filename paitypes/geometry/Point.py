import math
from typing import Iterator

from dataclasses import dataclass

from paitypes.geometry.Shape import Shape


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __str__(self) -> str:
        return '(' + str(self.x) + ',' + str(self.y) + ')'

    def distance_to_point(self, other_point: 'Point') -> float:
        x_distance = (self.x - other_point.x) ** 2
        y_distance = (self.y - other_point.y) ** 2
        euclidean_distance = math.sqrt(x_distance + y_distance)
        return euclidean_distance

    def translate(self, translation: Shape) -> 'Point':
        return Point(self.x + translation.width, self.y + translation.height)


@dataclass(frozen=True)
class PointEstimate(Point):
    confidence: float

    def __str__(self) -> str:
        return '(({},{}) conf {})'.format(
            str(self.x),
            str(self.y),
            str(self.confidence)
        )

    def __iter__(self) -> Iterator[float]:
        yield from [getattr(self, x) for x in vars(self)]
