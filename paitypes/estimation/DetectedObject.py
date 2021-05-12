from enum import Enum
from typing import Dict

from paitypes.geometry.bounding_box.BoundingBox import BoundingBox
from dataclasses import dataclass


class Label(Enum):
    UNKNOWN = 0
    HUMAN = 1
    SEATBELT = 2


@dataclass
class DetectedObject:
    ID: int
    bounding_box: BoundingBox
    label: Label
    confidence: float
    verified: bool = True

    def __hash__(self) -> int:
        return hash((self.ID, self.bounding_box, self.confidence, self.label))
