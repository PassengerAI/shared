import numpy as np
from dataclasses import dataclass
from paitypes.geometry.Point import Point


@dataclass
class Vector:
    x: float
    y: float
    z: float = 0

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector') -> 'Vector':
        return self + (-other)

    def __neg__(self) -> 'Vector':
        return Vector(-self.x, -self.y, -self.z)

    def cross(self, other: 'Vector') -> 'Vector':
        return Vector.from_np_array(
            np.cross(self.to_np_array(), other.to_np_array()))

    def dot(self, other: 'Vector') -> float:
        return float(np.dot(self.to_np_array(), other.to_np_array()))

    def to_np_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_np_array(cls, nparray: np.ndarray) -> 'Vector':
        return cls(x=nparray[0], y=nparray[1], z=nparray[2])

    @classmethod
    def from_points(cls, p_start: Point, p_end: Point) -> 'Vector':
        return cls(x=p_end.x - p_start.x, y=p_end.y - p_start.y, z=0)
