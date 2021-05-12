from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
# TODO: Generically support `int`, `float`, etc.
class Shape:
    width: float
    height: float

    @classmethod
    def from_image(cls, img: np.ndarray) -> 'Shape':
        height, width = img.shape[:2]
        return cls(height=height, width=width)

    def to_cv2(self) -> Tuple[int, int]:
        return self.to_numpy()[::-1]

    def to_numpy(self) -> Tuple[int, int]:
        return (int(self.height), int(self.width))
