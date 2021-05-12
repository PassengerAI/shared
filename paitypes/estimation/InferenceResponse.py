from typing import List, Optional

from dataclasses import dataclass

from paitypes.estimation.InferenceResult import (PoseEstimationResult,
                                                 ObjectDetectionResult)


@dataclass
class InferenceResponse:
    pose_estimation_results: Optional[List[PoseEstimationResult]] = None
    object_detections_results: Optional[List[ObjectDetectionResult]] = None
    error: Optional[Exception] = None
