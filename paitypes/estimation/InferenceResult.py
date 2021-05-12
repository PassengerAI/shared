from typing import List

from paitypes.estimation.DetectedObject import DetectedObject
from paitypes.estimation.pose import Human

PoseEstimationResult = List[Human]
ObjectDetectionResult = List[DetectedObject]
